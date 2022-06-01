import os
import pickle
import argparse
from transformers import RobertaTokenizer
from mspan_roberta_gcn.drop_roberta_dataset import DropReader
from tag_mspan_robert_gcn.drop_roberta_mspan_dataset import DropReader as TDropReader

parser = argparse.ArgumentParser()
parser.add_argument("--data_type", type=str)
parser.add_argument("--input_path", type=str)
parser.add_argument("--roberta_path", type=str)
parser.add_argument("--output_dir", type=str)
parser.add_argument("--passage_length_limit", type=int, default=463)
parser.add_argument("--question_length_limit", type=int, default=46)
parser.add_argument("--tag_mspan", action="store_true")
parser.add_argument("--mode", type=str, default=None)

args = parser.parse_args()

tokenizer = RobertaTokenizer.from_pretrained(args.roberta_path)

if args.tag_mspan:
    dev_reader = TDropReader(
        tokenizer, args.passage_length_limit, args.question_length_limit
    )

    train_reader = TDropReader(
        tokenizer,
        args.passage_length_limit,
        args.question_length_limit,
        skip_when_all_empty=[
            "passage_span",
            "question_span",
            "addition_subtraction",
            "counting",
            "multi_span",
        ],
    )

    data_format = "{}_dataset_{}.json"

    if args.mode is None:
        data_mode = ["train"]
        for dm in data_mode:
            dpath = os.path.join(args.input_path, data_format.format(args.data_type, dm))
            data = train_reader._read(dpath)
            print(
                "Save data to {}.".format(
                    os.path.join(args.output_dir, "tmspan_cached_roberta_{}_{}.pkl".format(args.data_type, dm))
                )
            )
            with open(
                os.path.join(args.output_dir, "tmspan_cached_roberta_{}_{}.pkl".format(args.data_type, dm)),
                "wb",
            ) as f:
                # pickle fails to load this large data as one chunck
                if args.data_type == 'textual':
                    N = len(data) // 3
                    print('Saving 1/3 of data...')
                    pickle.dump(data[:N], f)
                    print('Saving 2/3 of data...')
                    pickle.dump(data[N:2*N], f)
                    print('Saving 3/3 of data...')
                    pickle.dump(data[2*N:], f)
                else:
                    pickle.dump(data, f)

        data_mode = ["dev"]
        for dm in data_mode:
            dpath = os.path.join(args.input_path, data_format.format(args.data_type, dm))
            data = dev_reader._read(dpath) if dm == "dev" else train_reader._read(dpath)
            print(
                "Save data to {}.".format(
                    os.path.join(args.output_dir, "tmspan_cached_roberta_{}_{}.pkl".format(args.data_type, dm))
                )
            )
            with open(
                os.path.join(args.output_dir, "tmspan_cached_roberta_{}_{}.pkl".format(args.data_type, dm)),
                "wb",
            ) as f:
                pickle.dump(data, f)
    else:
        data_mode = args.mode
        dpath = os.path.join(args.input_path, data_format.format(args.data_type, data_mode))
        data = train_reader._read(dpath) if data_mode == "train" else dev_reader._read(dpath)
        print(
            "Save data to {}.".format(
                os.path.join(args.output_dir, "tmspan_cached_roberta_{}_{}.pkl".format(args.data_type, data_mode))
            )
        )
        with open(
            os.path.join(args.output_dir, "tmspan_cached_roberta_{}_{}.pkl".format(args.data_type, data_mode)),
            "wb",
        ) as f:
            pickle.dump(data, f)
else:
    dev_reader = DropReader(
        tokenizer, args.passage_length_limit, args.question_length_limit
    )
    train_reader = DropReader(
        tokenizer,
        args.passage_length_limit,
        args.question_length_limit,
        skip_when_all_empty=[
            "passage_span",
            "question_span",
            "addition_subtraction",
            "counting",
        ],
    )

    data_format = "{}_dataset_{}.json"

    data_mode = ["train"]
    for dm in data_mode:
        dpath = os.path.join(args.input_path, data_format.format(args.data_type, dm))
        data = train_reader._read(dpath)
        print(
            "Save data to {}.".format(
                os.path.join(args.output_dir, "cached_roberta_{}_{}.pkl".format(args.data_type, dm))
            )
        )
        with open(
            os.path.join(args.output_dir, "cached_roberta_{}_{}.pkl".format(args.data_type, dm)), "wb"
        ) as f:
            if args.data_type == 'textual':
                N = len(data) // 3
                print('Saving 1/3 of data...')
                pickle.dump(data[:N], f)
                print('Saving 2/3 of data...')
                pickle.dump(data[N:2*N], f)
                print('Saving 3/3 of data...')
                pickle.dump(data[2*N:], f)
            else:
                pickle.dump(data, f)

    data_mode = ["dev"]
    for dm in data_mode:
        dpath = os.path.join(args.input_path, data_format.format(args.data_type, dm))
        data = dev_reader._read(dpath) if dm == "dev" else train_reader._read(dpath)
        print(
            "Save data to {}.".format(
                os.path.join(args.output_dir, "cached_roberta_{}_{}.pkl".format(args.data_type, dm))
            )
        )
        with open(
            os.path.join(args.output_dir, "cached_roberta_{}_{}.pkl".format(args.data_type, dm)), "wb"
        ) as f:
            pickle.dump(data, f)
