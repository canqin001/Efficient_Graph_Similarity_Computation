from utils import tab_printer
from egsc_nonkd import EGSC_NonKD_Trainer
from parser import parameter_parser


def main():
    """
    Parsing command line parameters, reading data, fitting and scoring a SimGNN model.
    """
    args = parameter_parser()
    tab_printer(args)
    trainer = EGSC_NonKD_Trainer(args)
    
    trainer.fit()
    trainer.score()
    
    if args.notify:
        import os
        import sys
        if sys.platform == 'linux':
            os.system('notify-send SimGNN "Program is finished."')
        elif sys.platform == 'posix':
            os.system("""
                      osascript -e 'display notification "SimGNN" with title "Program is finished."'
                      """)
        else:
            raise NotImplementedError('No notification support for this OS.')


if __name__ == "__main__":
    main()
