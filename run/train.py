import sys
sys.path.remove('/home/charon/project/sensorium/run')
sys.path.append('/home/charon/project')
from shallowmind.api.train import train

if __name__ == '__main__':
    train()