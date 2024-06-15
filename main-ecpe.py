import argparse
import os
import logging
import time
import pickle
from tqdm import tqdm
import jieba

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import seed_everything

# from transformers import MT5ForConditionalGeneration, T5Tokenizer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import AdamW, get_linear_schedule_with_warmup
# from transformers import BertTokenizer, MT5ForConditionalGeneration

from data_utils_fe_and_emotion import SINADataset, NTCIRDataset
from data_utils_fe_and_emotion import write_results_to_log, read_sina_line_examples_from_file
from eval_utils_fe_and_emotion import compute_scores
from os.path import join

logger = logging.getLogger(__name__)


def init_args():
    parser = argparse.ArgumentParser()
    # basic settings
    parser.add_argument("--task", default='ecpe', type=str, required=True,
                        help="The name of the task, selected from: [uabsa, aste, tasd, aope, ecpe, ee, ce, alle]")
    parser.add_argument("--dataset", default='eca_ch', type=str, required=True,
                        help="The name of the dataset, selected from: [laptop14, rest14, rest15, rest16, eca_ch]")
    parser.add_argument("--model_name_or_path", default='google/mt5-base', type=str,
                        help="Path to pre-trained model or shortcut name")
    parser.add_argument("--paradigm", default='annotation', type=str, required=True,
                        help="The way to construct target sentence, selected from: [annotation, extraction]")
    parser.add_argument("--do_train", action='store_true', help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true', help="Whether to run eval on the dev/test set.")
    parser.add_argument("--do_direct_eval", action='store_true',
                        help="Whether to run direct eval on the dev/test set.")

    # Other parameters
    parser.add_argument("--max_seq_length", default=30, type=int)   ## ch:128 en:30
    parser.add_argument("--n_gpu", default=0)
    parser.add_argument("--train_batch_size", default=16, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=16, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=3e-4, type=float)    ## eng:0.005
    parser.add_argument("--num_train_epochs", default=20, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument('--seed', type=int, default=2024, help="random seed for initialization")

    # training details
    parser.add_argument("--weight_decay", default=0.0, type=float)
    parser.add_argument("--adam_epsilon", default=1e-8, type=float)
    parser.add_argument("--warmup_steps", default=0.0, type=float)

    args = parser.parse_args()

    # set up output dir which looks like './aste/rest14/extraction/'
    if not os.path.exists('./outputs'):
        os.mkdir('./outputs')

    task_dir = f"./outputs/{args.task}"
    if not os.path.exists(task_dir):
        os.mkdir(task_dir)

    task_dataset_dir = f"{task_dir}/{args.dataset}"
    if not os.path.exists(task_dataset_dir):
        os.mkdir(task_dataset_dir)

    output_dir = f"{task_dataset_dir}/{args.paradigm}"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    args.output_dir = output_dir

    return args


def get_dataset(tokenizer, fold_id, type_path, args):
    if args.dataset == 'eca_ch':
        return SINADataset(tokenizer=tokenizer, fold_id=fold_id, data_type=type_path, task=args.task,
                           paradigm=args.paradigm, data_dir=args.dataset)
    elif args.dataset == 'eca_eng':
        return NTCIRDataset(tokenizer=tokenizer, fold_id=fold_id, data_type=type_path, task=args.task,
                           paradigm=args.paradigm, data_dir=args.dataset)


# class T5PegasusTokenizer(BertTokenizer):
#     """结合中文特点完善的Tokenizer
#     基于词颗粒度的分词，如词表中未出现，再调用BERT原生Tokenizer
#     """
#     def __init__(self, pre_tokenizer=lambda x: jieba.cut(x, HMM=False), *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.pre_tokenizer = pre_tokenizer
#
#     def _tokenize(self, text, *arg, **kwargs):
#         split_tokens = []
#         for text in self.pre_tokenizer(text):
#             if text in self.vocab:
#                 split_tokens.append(text)
#             else:
#                 split_tokens.extend(super()._tokenize(text))
#         return


class T5FineTuner(pl.LightningModule):
    def __init__(self, hparams, fold_id):
        print('hparams:', hparams)
        super(T5FineTuner, self).__init__()
        self.fold_id = fold_id
        self.hparams = hparams
        # self.model = MT5ForConditionalGeneration.from_pretrained(hparams.model_name_or_path)
        # self.tokenizer = T5Tokenizer.from_pretrained(hparams.model_name_or_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(hparams.model_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(hparams.model_name_or_path)

    def is_logger(self):
        return True

    def forward(self, input_ids, attention_mask=None, decoder_input_ids=None,
                decoder_attention_mask=None, labels=None):
        return self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels,
        )

    def _step(self, batch):
        lm_labels = batch["target_ids"]
        lm_labels[lm_labels[:, :] == self.tokenizer.pad_token_id] = -100

        outputs = self(
            input_ids=batch["source_ids"],
            attention_mask=batch["source_mask"],
            labels=lm_labels,
            decoder_attention_mask=batch['target_mask']
        )

        loss = outputs[0]
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._step(batch)

        tensorboard_logs = {"train_loss": loss}
        return {"loss": loss, "log": tensorboard_logs}

    def training_epoch_end(self, outputs):
        avg_train_loss = torch.stack([x["loss"] for x in outputs]).mean()
        tensorboard_logs = {"avg_train_loss": avg_train_loss}
        return {"avg_train_loss": avg_train_loss, "log": tensorboard_logs, 'progress_bar': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        loss = self._step(batch)
        return {"val_loss": loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        tensorboard_logs = {"val_loss": avg_loss}
        return {"avg_val_loss": avg_loss, "log": tensorboard_logs, 'progress_bar': tensorboard_logs}

    def configure_optimizers(self):
        '''Prepare optimizer and schedule (linear warmup and decay)'''
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)
        self.opt = optimizer
        return [optimizer]

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx, second_order_closure=None):
        if self.trainer.use_tpu:
            xm.optimizer_step(optimizer)
        else:
            optimizer.step()
        optimizer.zero_grad()
        self.lr_scheduler.step()

    def get_tqdm_dict(self):
        tqdm_dict = {"loss": "{:.4f}".format(self.trainer.avg_loss), "lr": self.lr_scheduler.get_last_lr()[-1]}
        return tqdm_dict

    def train_dataloader(self):
        train_dataset = get_dataset(tokenizer=self.tokenizer, fold_id=self.fold_id, type_path="train",
                                    args=self.hparams)
        dataloader = DataLoader(train_dataset, batch_size=self.hparams.train_batch_size, drop_last=True, shuffle=True,
                                num_workers=4)
        t_total = (
                (len(dataloader.dataset) // (self.hparams.train_batch_size * max(1, len(self.hparams.n_gpu))))
                // self.hparams.gradient_accumulation_steps
                * float(self.hparams.num_train_epochs)
        )
        scheduler = get_linear_schedule_with_warmup(
            self.opt, num_warmup_steps=self.hparams.warmup_steps, num_training_steps=t_total
        )
        self.lr_scheduler = scheduler
        return dataloader

    def val_dataloader(self):
        val_dataset = get_dataset(tokenizer=self.tokenizer, fold_id=self.fold_id, type_path="test", args=self.hparams)
        return DataLoader(val_dataset, batch_size=self.hparams.eval_batch_size, num_workers=4)


class LoggingCallback(pl.Callback):
    def on_validation_end(self, trainer, pl_module):
        logger.info("***** Validation results *****")
        if pl_module.is_logger():
            metrics = trainer.callback_metrics
        # Log results
        for key in sorted(metrics):
            if key not in ["log", "progress_bar"]:
                logger.info("{} = {}\n".format(key, str(metrics[key])))

    def on_test_end(self, trainer, pl_module):
        logger.info("***** Test results *****")

        if pl_module.is_logger():
            metrics = trainer.callback_metrics

        # Log and save results to file
        output_test_results_file = os.path.join(pl_module.hparams.output_dir, "test_results.txt")
        with open(output_test_results_file, "w") as writer:
            for key in sorted(metrics):
                if key not in ["log", "progress_bar"]:
                    logger.info("{} = {}\n".format(key, str(metrics[key])))
                    writer.write("{} = {}\n".format(key, str(metrics[key])))


def evaluate(args, tokenizer, data_loader, model, paradigm, task, sents):
    """
    Compute scores given the predictions and gold labels
    """
    # device = torch.device(f'cuda:{args.n_gpu}')
    device = torch.device(f'cuda:{args.n_gpu}' if torch.cuda.is_available() else 'cpu')

    model.model.to(device)

    model.model.eval()
    outputs, targets = [], []
    for batch in tqdm(data_loader):
        # need to push the data to device
        outs = model.model.generate(input_ids=batch['source_ids'].to(device),
                                    attention_mask=batch['source_mask'].to(device),
                                    max_length=512)

        dec = [tokenizer.decode(ids, skip_special_tokens=True) for ids in outs]
        target = [tokenizer.decode(ids, skip_special_tokens=True) for ids in batch["target_ids"]]


        outputs.extend(dec)
        targets.extend(target)

    if task in ['ee', 'ce', 'ece']:
        # print('targets:', targets)
        raw_scores, fixed_scores, all_labels, all_preds, all_preds_fixed = compute_scores(outputs, targets, sents, paradigm,
                                                                                          task)
        results = {'raw_scores': raw_scores, 'fixed_scores': fixed_scores, 'labels': all_labels,
                   'preds': all_preds, 'preds_fixed': all_preds_fixed}
        # pickle.dump(results, open(f"{args.output_dir}/results-{args.task}-{args.dataset}-{args.paradigm}.pickle", 'wb'))
        return raw_scores, fixed_scores
    elif task in ['ecpe']:
        # print('targets:', targets)
        raw_scores_ecpe, raw_scores_ee, raw_scores_ce, fixed_scores_ecpe, fixed_scores_ee, fixed_scores_ce, \
        all_labels, all_predictions, all_predictions_fixed = compute_scores(outputs, targets, sents, paradigm, task)
        results = {'raw_scores_ecpe': raw_scores_ecpe, 'fixed_scores_ecpe': fixed_scores_ecpe,
                   'raw_scores_ee': raw_scores_ee, 'fixed_scores_ee': fixed_scores_ee,
                   'raw_scores_ce': raw_scores_ce, 'fixed_scores_ce': fixed_scores_ce,
                   'labels': all_labels,
                   'preds': all_predictions, 'preds_fixed': all_predictions_fixed}
        return raw_scores_ecpe, raw_scores_ee, raw_scores_ce, fixed_scores_ecpe, fixed_scores_ee, fixed_scores_ce


def main():
    # initialization
    print('-'*10, 'initialization', '-'*10)
    args = init_args()
    print("\n", "=" * 30, f"NEW EXP: {args.task.upper()} on {args.dataset}", "=" * 30, "\n")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    seed_everything(args.seed)

    # tokenizer = T5Tokenizer.from_pretrained(args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    for fold_id in range(6, 11):
        print('-'*15, 'fold{}'.format(fold_id), '-'*15)

        # show one sample to check the sanity of the code and the expected output
        print(f"Here is an example (from dev set) under `{args.paradigm}` paradigm:")
        if args.dataset == 'eca_ch':
            dataset = SINADataset(tokenizer=tokenizer, fold_id=fold_id, data_type='test', task=args.task,
                                  paradigm=args.paradigm, data_dir=args.dataset)
        elif args.dataset == 'eca_eng':
            dataset = NTCIRDataset(tokenizer=tokenizer, fold_id=fold_id, data_type='test', task=args.task,
                                  paradigm=args.paradigm, data_dir=args.dataset)

        # dataset = ABSADataset(tokenizer=tokenizer, data_dir=args.dataset, data_type='dev',
        #                       paradigm=args.paradigm, task=args.task, max_len=args.max_seq_length)

        data_sample = dataset[6]  # a random data sample
        print('Input :', tokenizer.decode(data_sample['source_ids'], skip_special_tokens=True))
        print('Output:', tokenizer.decode(data_sample['target_ids'], skip_special_tokens=True))

        # training process
        if args.do_train:
            print("\n****** Conduct Training ******")
            model = T5FineTuner(args, fold_id=fold_id)

            checkpoint_callback = pl.callbacks.ModelCheckpoint(
                filepath=args.output_dir, prefix="ckt", monitor='val_loss', mode='min', save_top_k=3
            )

            # prepare for trainer
            train_params = dict(
                default_root_dir=args.output_dir,
                accumulate_grad_batches=args.gradient_accumulation_steps,
                # gpus=args.n_gpu,
                gradient_clip_val=1.0,
                amp_level='O1',
                max_epochs=args.num_train_epochs,
                checkpoint_callback=checkpoint_callback,
                callbacks=[LoggingCallback()],
            )

            trainer = pl.Trainer(**train_params)
            trainer.fit(model)

            # save the final model
            # model.model.save_pretrained(args.output_dir)

            print("Finish training and saving the model!")

        if args.do_eval:

            print("\n****** Conduct Evaluating ******")

            # model = T5FineTuner(args)
            dev_results, test_results = {}, {}
            best_f1, best_checkpoint, best_epoch = -999999.0, None, None
            all_checkpoints, all_epochs = [], []

            # retrieve all the saved checkpoints for model selection
            saved_model_dir = args.output_dir
            for f in os.listdir(saved_model_dir):
                file_name = os.path.join(saved_model_dir, f)
                if 'cktepoch' in file_name:
                    all_checkpoints.append(file_name)

            # conduct some selection (or not)
            print(f"We will perform validation on the following checkpoints: {all_checkpoints}")

            # load dev and test datasets
            dev_dataset = SINADataset(tokenizer, fold_id=fold_id, data_type='dev', task=args.task,
                                      paradigm=args.paradigm, data_dir=args.dataset)
            dev_loader = DataLoader(dev_dataset, batch_size=32, num_workers=4)

            test_dataset = SINADataset(tokenizer, fold_id=fold_id, data_type='test', task=args.task,
                                       paradigm=args.paradigm, data_dir=args.dataset)
            test_loader = DataLoader(test_dataset, batch_size=32, num_workers=4)

            for checkpoint in all_checkpoints:
                epoch = checkpoint.split('=')[-1][:-5] if len(checkpoint) > 1 else ""
                # only perform evaluation at the specific epochs ("15-19")
                # eval_begin, eval_end = args.eval_begin_end.split('-')
                if 0 <= int(epoch) < 100:
                    all_epochs.append(epoch)

                    # reload the model and conduct inference
                    print(f"\nLoad the trained model from {checkpoint}...")
                    model_ckpt = torch.load(checkpoint)
                    model = T5FineTuner(model_ckpt['hyper_parameters'])
                    model.load_state_dict(model_ckpt['state_dict'])

                    dev_result = evaluate(dev_loader, model, args.paradigm, args.task)
                    if dev_result['f1'] > best_f1:
                        best_f1 = dev_result['f1']
                        best_checkpoint = checkpoint
                        best_epoch = epoch

                    # add the global step to the name of these metrics for recording
                    # 'f1' --> 'f1_1000'
                    dev_result = dict((k + '_{}'.format(epoch), v) for k, v in dev_result.items())
                    dev_results.update(dev_result)

                    test_result = evaluate(test_loader, model, args.paradigm, args.task)
                    test_result = dict((k + '_{}'.format(epoch), v) for k, v in test_result.items())
                    test_results.update(test_result)

            # print test results over last few steps
            print(f"\n\nThe best checkpoint is {best_checkpoint}")
            best_step_metric = f"f1_{best_epoch}"
            print(f"F1 scores on test set: {test_results[best_step_metric]:.4f}")

            print("\n* Results *:  Dev  /  Test  \n")
            metric_names = ['f1', 'precision', 'recall']
            for epoch in all_epochs:
                print(f"Epoch-{epoch}:")
                for name in metric_names:
                    name_step = f'{name}_{epoch}'
                    print(f"{name:<10}: {dev_results[name_step]:.4f} / {test_results[name_step]:.4f}", sep='  ')
                print()

            results_log_dir = './results_log'
            if not os.path.exists(results_log_dir):
                os.mkdir(results_log_dir)
            log_file_path = f"{results_log_dir}/{args.task}-{args.dataset}.txt"
            write_results_to_log(log_file_path, test_results[best_step_metric], args, dev_results, test_results, all_epochs)

        # evaluation process
        if args.do_direct_eval:
            print("\n****** Conduct Evaluating with the last state ******")

            # model = T5FineTuner(args)

            # print("Reload the model")
            # model.model.from_pretrained(args.output_dir)
            TEST_FILE = 'fold%s_test.txt'

            data_path = join('data/' + args.task + '/' + args.dataset, TEST_FILE % fold_id)
            doc_ids, doc_lens, doc_sents, doc_ecps, doc_emos, doc_caus, doc_emocat = read_sina_line_examples_from_file(data_path)

            if args.dataset == 'eca_ch':
                test_dataset = SINADataset(tokenizer, fold_id=fold_id, data_type='test', task=args.task,
                                           paradigm=args.paradigm, data_dir=args.dataset)
            elif args.dataset == 'eca_eng':
                test_dataset = NTCIRDataset(tokenizer, fold_id=fold_id, data_type='test', task=args.task,
                                           paradigm=args.paradigm, data_dir=args.dataset)
            test_loader = DataLoader(test_dataset, batch_size=32, num_workers=4)
            # print(test_loader.device)
            if args.task in ['ee', 'ce', 'ece']:
                raw_scores, fixed_scores = \
                    evaluate(args, tokenizer, test_loader, model, args.paradigm, args.task, doc_sents)

                # write to file
                log_file_path = f"results_log/{args.task}-{args.dataset}.txt"
                local_time = time.asctime(time.localtime(time.time()))
                exp_settings = f"{args.task} on {args.dataset} under {args.paradigm}; " \
                               f"fold_id = {fold_id}," \
                               f"Train bs={args.train_batch_size}, " \
                               f"num_epochs = {args.num_train_epochs}"
                exp_results = f"Raw P={raw_scores['precision']:.4f}, R={raw_scores['recall']:.4f}, F1 = {raw_scores['f1']:.4f}, " \
                              f"Fixed P={fixed_scores['precision']:.4f}, R={fixed_scores['recall']:.4f}, F1 = {fixed_scores['f1']:.4f}"
                log_str = f'============================================================\n'
                log_str += f"{local_time}\n{exp_settings}\n{exp_results}\n\n"
                with open(log_file_path, "a+") as f:
                    f.write(log_str)

            elif args.task in ['ecpe']:
                raw_scores_ecpe, raw_scores_ee, raw_scores_ce, fixed_scores_ecpe, fixed_scores_ee, fixed_scores_ce = \
                    evaluate(args, tokenizer, test_loader, model, args.paradigm, args.task, doc_sents)

                # write to file
                log_file_path = f"results_log/{args.task}-{args.dataset}.txt"
                local_time = time.asctime(time.localtime(time.time()))
                exp_settings = f"{args.task} on {args.dataset} under {args.paradigm}; " \
                               f"fold_id = {fold_id},"\
                               f"Train bs={args.train_batch_size}, " \
                               f"num_epochs = {args.num_train_epochs}"
                exp_results = f"Raw P={raw_scores_ecpe['precision']:.4f}, R={raw_scores_ecpe['recall']:.4f}, F1 = {raw_scores_ecpe['f1']:.4f}, " \
                              f"Fixed P={fixed_scores_ecpe['precision']:.4f}, R={fixed_scores_ecpe['recall']:.4f}, F1 = {fixed_scores_ecpe['f1']:.4f}, " \
                              f"Raw P={raw_scores_ee['precision']:.4f}, R={raw_scores_ee['recall']:.4f}, F1 = {raw_scores_ee['f1']:.4f}, " \
                              f"Fixed P={fixed_scores_ee['precision']:.4f}, R={fixed_scores_ee['recall']:.4f}, F1 = {fixed_scores_ee['f1']:.4f}, " \
                              f"Raw P={raw_scores_ce['precision']:.4f}, R={raw_scores_ce['recall']:.4f}, F1 = {raw_scores_ce['f1']:.4f}, " \
                              f"Fixed P={fixed_scores_ce['precision']:.4f}, R={fixed_scores_ce['recall']:.4f}, F1 = {fixed_scores_ce['f1']:.4f}"

                log_str = f'============================================================\n'
                log_str += f"{local_time}\n{exp_settings}\n{exp_results}\n\n"
                with open(log_file_path, "a+") as f:
                    f.write(log_str)


if __name__ == '__main__':
    main()
