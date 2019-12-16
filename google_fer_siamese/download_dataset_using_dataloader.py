from dataset import SiameseGoogleFer
import time
from torchvision import datasets, transforms

t = time.time()
transform = transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])
train_path = "data/faceexp-comparison-data-train-public.csv"
test_path = "data/faceexp-comparison-data-test-public.csv"
#train_dataset = SiameseGoogleFer(train_path, train_flag=True, transform=transform)
test_dataset = SiameseGoogleFer(test_path, train_flag=False, transform=transform)
print(time.time()-t)
