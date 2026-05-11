import sys

from fitzpatrick_optimizer.train import main


if __name__ == "__main__":
    argv = ["--model", "residual-filter", *sys.argv[1:]]
    if "--scale_dataset" in argv:
        index = argv.index("--scale_dataset")
        argv[index] = "--max_samples"
    main(argv)
