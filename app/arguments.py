import argparse

def arguments():
    parser = argparse.ArgumentParser()

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-t', '--train', action='store_true')
    group.add_argument('-e', '--eval', action='store_true')

    arguments = parser.parse_args()

    if arguments.train:
        return True
    elif arguments.eval:
        return False 