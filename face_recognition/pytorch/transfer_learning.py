import torch
from torch.utils.data import DataLoader
from torch import optim
import torchvision.transforms as ttf
from dataloader import lfwCustom
from facenet_pytorch import InceptionResnetV1

from tqdm import tqdm
import argparse
import wandb

def main(args):
    data_dir = "./lfw/lfwcrop_color/faces"

    train_transforms = ttf.Compose([ttf.RandomResizedCrop((224, 224), scale=(0.1, 1), ratio=(0.5, 2)), \
            ttf.GaussianBlur(3, sigma=(0.1, 2.0)), ttf.RandomHorizontalFlip(), \
            ttf.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5), ttf.ToTensor()])
    # train_transforms = ttf.Compose([ttf.Resize((224, 224)), ttf.GaussianBlur(3, sigma=(0.1, 2.0)), ttf.RandomHorizontalFlip(), \
    #                    ttf.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5), ttf.ToTensor()])
    test_transforms = ttf.Compose([ttf.Resize((224, 224)), ttf.ToTensor()])
    train_loader = DataLoader(lfwCustom(data_dir, transforms=train_transforms), shuffle=True, num_workers=2, batch_size=args.batch_size)
    test_loader = DataLoader(lfwCustom(data_dir, transforms=test_transforms, train=False), shuffle=False, num_workers=1, batch_size=args.batch_size)

    model = InceptionResnetV1(classify=True, pretrained='vggface2', num_classes=5749).to(args.device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.9)
    # criterion = torch.nn.CrossEntropyLoss(reduction='none')
    criterion = torch.nn.CrossEntropyLoss()

    best_loss = 100

    for epoch in range(args.epochs):
        print("Epoch {}/{}".format((epoch+1), args.epochs))
        model.train()
    
        train_loss = 0
        num_correct = 0

        batch_bar = tqdm(total=len(train_loader), dynamic_ncols=True, leave=False, position=0, desc='Train')
        for i, (img, label) in enumerate(train_loader):
            optimizer.zero_grad()
            img = img.to(args.device)
            label = label.to(args.device)
            
            outputs = model(img)
            loss = criterion(outputs, label)
            train_loss += loss.item()

            try: 
                curr_lr = scheduler.get_lr()[0]
            except:
                curr_lr = scheduler.get_lr()

            if args.wandb == True:
                wandb.log({"train loss": loss})
                wandb.log({"learning rate": curr_lr})

            num_correct += int((torch.argmax(outputs, axis=1) == label).sum())

            batch_bar.set_postfix(loss="{:.04f}".format(loss.item()), num_correct="{}".format(num_correct), lr="{:.06f}".format(optimizer.param_groups[0]['lr']))

            loss.backward()
            optimizer.step()
            scheduler.step()

            batch_bar.update() 
        batch_bar.close()
        print("Train Loss {:.04f}, Learning rate {:.06f}".format(float(train_loss/len(train_loader)), float(optimizer.param_groups[0]['lr'])))

        # Validation
        test_loss = 0
        for i, (img, label) in enumerate(test_loader):
            model.eval()
            batch_bar = tqdm(total=len(test_loader), dynamic_ncols=True, leave=False, position=0, desc='Validation')

            img = img.to(args.device)
            label = label.to(args.device)

            with torch.no_grad():
                outputs = model(img)

            loss = criterion(outputs, label)
            test_loss += loss.item()

            wandb.log({"val loss": loss})

            batch_bar.set_postfix(loss="{:.04f}".format(loss.item()))
            batch_bar.update()  
        batch_bar.close() 
        print("Valid Loss {:.04f}".format(float(test_loss/len(test_loader))))

        # Model save
        torch.save(model.state_dict(), './saved_models/curr_model_{:02f}.pth'.format(float(test_loss/len(test_loader))))
        if test_loss < best_loss:
            torch.save(model.state_dict(), './saved_models/val_{:.02f}.pth'.format(float(test_loss/len(test_loader))))
            best_loss = test_loss
            print("--- best model saved at ./outputs ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0", type=str, help="Select cuda:0 or cuda:1")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--test_split", help="train test split size", type=float, default=0.2)
    parser.add_argument("--lr", help="learning rate", type=float, default=1e-4)
    parser.add_argument("--epochs", help="Number of epochs to train", type=int, default=50)
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args()

    if args.wandb == True:    
        project_name = 'cmu-face-recognition'
        wandb.init(project=project_name, entity="juntae9926")
        wandb.config.update(args)
        print(f"Start with wandb with {project_name}")
    main(args)