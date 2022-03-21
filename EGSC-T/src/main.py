from utils import tab_printer
from egsc import EGSCTrainer
from parser import parameter_parser


def main():
    args = parameter_parser()
    tab_printer(args)
    trainer = EGSCTrainer(args)
    
    trainer.fit()
    trainer.score()
    trainer.save_model()
    
    if args.notify:
        import os
        import sys
        if sys.platform == 'linux':
            os.system('notify-send EGSC "Program is finished."')
        elif sys.platform == 'posix':
            os.system("""
                      osascript -e 'display notification "EGSC" with title "Program is finished."'
                      """)
        else:
            raise NotImplementedError('No notification support for this OS.')


if __name__ == "__main__":
    main()
