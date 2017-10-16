HYPER_PARAMS = None

def initialise_arguments(args_parser):

    args_parser.add_argument(
        '--train-files',
        help='GCS or local paths to training data',
        nargs='+',
        required=True
    )
    args_parser.add_argument(
        '--num-epochs',
        help="""\
        Maximum number of training data epochs on which to train.
        If both --max-steps and --num-epochs are specified,
        the training job will run for --max-steps or --num-epochs,
        whichever occurs first. If unspecified will run for --max-steps.\
        """,
        default=1000,
        type=int,
    )
    args_parser.add_argument(
        '--train-batch-size',
        help='Batch size for training steps',
        type=int,
        default=1000
    )

    args_parser.add_argument(
        '--eval-files',
        help='GCS or local paths to evaluation data',
        nargs='+',
        required=True
    )

    args_parser.add_argument(
        '--eval-batch-size',
        help='Batch size for evaluation steps',
        type=int,
        default=1000
    )

    # Pre-processing arguments
    args_parser.add_argument(
        '--embedding-size',
        help='Number of embedding dimensions for categorical columns. value of 0 means no embedding',
        default=0,
        type=int
    )

    # Estimator arguments
    args_parser.add_argument(
        '--learning-rate',
        help="Learning rate value for the optimizers",
        default=0.001,
        type=float
    )
    args_parser.add_argument(
        '--hidden-units',
        help="""\
             Hidden layer sizes to use for DNN feature columns, provided in comma-separated layers. 
             If --scale-factor > 0, then only the size of the first layer will be used to compute the sizes of subsquent layers \
             """,
        default='128,32,4'
    )
    args_parser.add_argument(
        '--layer-sizes-scale-factor',
        help="""\
            Determine how the size of the layers in the DNN decays. 
            If value = 0 then the provided --hidden-units will be taken as is\
            """,
        default=0.7,
        type=float
    )
    args_parser.add_argument(
        '--num-layers',
        help='Number of layers in the DNN. If --scale-factor > 0, then this parameter is ignored',
        default=2,
        type=int
    )
    args_parser.add_argument(
        '--dropout-prob',
        help="the probability we will drop out a given coordinate",
        default=None
    )
    args_parser.add_argument(
        '--job-dir',
        help='GCS location to write checkpoints and export models',
        required=True
    )

    # Experiment arguments
    args_parser.add_argument(
        '--eval-delay-secs',
        help='How long to wait before running first evaluation',
        default=10,
        type=int
    )
    args_parser.add_argument(
        '--min-eval-frequency',
        help='Minimum number of training steps between evaluations',
        default=None,  # Use TensorFlow's default (currently, 1000 on GCS)
        type=int
    )
    args_parser.add_argument(
        '--train-steps',
        help="""\
        Steps to run the training job for. If --num-epochs is not specified,
        this must be. Otherwise the training job will run indefinitely.\
        """,
        type=int
    )
    args_parser.add_argument(
        '--eval-steps',
        help='Number of steps to run evaluation for at each checkpoint',
        default=100,
        type=int
    )
    args_parser.add_argument(
        '--export-format',
        help='The input format of the exported SavedModel binary',
        choices=['JSON', 'CSV', 'EXAMPLE'],
        default='JSON'
    )

    # Argument to turn on all logging
    args_parser.add_argument(
        '--verbosity',
        choices=[
            'DEBUG',
            'ERROR',
            'FATAL',
            'INFO',
            'WARN'
        ],
        default='INFO',
    )

    args = args_parser.parse_args()

    return args

