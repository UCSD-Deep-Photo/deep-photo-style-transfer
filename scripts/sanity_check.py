import os

# Make sure basic usage such as training, testing, saving and loading
# runs without errors

def run(command):
    print(command)
    exit_status = os.system(command)
    if exit_status > 0:
        exit(1)


if __name__ == '__main__':
    pass