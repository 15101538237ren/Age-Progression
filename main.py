# -*- coding: utf-8
import model
import logging
from util import *
from option import parser
from torchvision.datasets.folder import pil_loader

if __name__ == '__main__':
    requirement_check()
    args = parser.parse_args()

    consts.NUM_Z_CHANNELS = args.z_channels
    net = model.Net()

    if not args.cpu and torch.cuda.is_available():
        net.cuda()

    if args.mode == 'train':

        betas = (args.b1, args.b2) if args.load is None else None
        weight_decay = args.weight_decay if args.load is None else None
        lr = args.learning_rate if args.load is None else None

        if args.load is not None:
            net.load(args.load)
            print("Loading pre-trained models from {}".format(args.load))

        data_src = args.input or consts.UTKFACE_DEFAULT_PATH
        print("Data folder is {}".format(data_src))
        results_dest = args.output or default_train_results_dir()
        os.makedirs(results_dest, exist_ok=True)
        print("Results folder is {}".format(results_dest))

        with open(os.path.join(results_dest, 'session_arguments.txt'), 'w') as info_file:
            info_file.write(' '.join(sys.argv))

        log_path = os.path.join(results_dest, 'log_results.log')
        if os.path.exists(log_path):
            os.remove(log_path)
        logging.basicConfig(filename=log_path, level=logging.DEBUG)

        net.teach(
            utkface_path=data_src,
            batch_size=args.batch_size,
            betas=betas,
            epochs=args.epochs,
            weight_decay=weight_decay,
            lr=lr,
            should_plot=args.sp,
            where_to_save=results_dest,
            models_saving=args.models_saving
        )
    elif args.mode == 'test':

        if args.load is None:
            raise RuntimeError("Must provide path of trained models")

        net.load(path=args.load, slim=True)

        results_dest = args.output or default_test_results_dir()
        if not os.path.isdir(results_dest):
            os.makedirs(results_dest)

        image_tensor = pil_to_model_tensor_transform(pil_loader(args.input)).to(net.device)
        net.test_single(
            image_tensor=image_tensor,
            age=args.age,
            gender=args.gender,
            target=results_dest,
            watermark=args.watermark
        )