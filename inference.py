transformation = transforms.Compose([
                                transforms.ToPILImage(),
                                transforms.Resize((70,70)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

#inference
import cv2
import numpy as np

model.eval()

img = cv2.imread("/content/5.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# img = cv2.resize(img,(28,28))
tensor = transformation(img).unsqueeze(0).to(device)

preds = model(tensor)

preds = preds.cpu().detach().numpy()
print(preds)
output = np.argmax(preds)
output
