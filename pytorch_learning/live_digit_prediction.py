import pygame
import torch
import torch.nn as nn
import numpy as np
from PIL import Image

class ImprovedMNIST(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.layer1  = nn.Linear(784, 256)
        self.bn1     = nn.BatchNorm1d(256)
        self.layer2  = nn.Linear(256, 128)
        self.bn2     = nn.BatchNorm1d(128)
        self.layer3  = nn.Linear(128, 64)
        self.bn3     = nn.BatchNorm1d(64)
        self.output  = nn.Linear(64, 10)
        self.relu    = nn.ReLU()

    def forward(self, x):
        x = self.flatten(x)
        x = self.relu(self.bn1(self.layer1(x)))
        x = self.relu(self.bn2(self.layer2(x)))
        x = self.relu(self.bn3(self.layer3(x)))
        x = self.output(x)
        return x


model = ImprovedMNIST()
model.load_state_dict(torch.load('mnist_model.pth'))
model.eval()

def preprocess(surface):
    data = pygame.surfarray.array3d(surface)
    data = data.transpose(1, 0, 2)          
    gray = np.mean(data, axis=2)            

    
    img  = Image.fromarray(gray.astype(np.uint8))
    img  = img.resize((28, 28), Image.LANCZOS)
    arr  = np.array(img).astype(np.float32)

    
    arr  = np.where(arr > 20, 255.0, 0.0).astype(np.float32)
    arr  = (arr / 127.5) - 1.0
    tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)
    return tensor

def predict(surface):
    tensor = preprocess(surface)
    with torch.no_grad():
        output = model(tensor)
        probs  = torch.softmax(output, dim=1)[0]
        top3   = torch.topk(probs, 3)
    return top3.indices.tolist(), top3.values.tolist()


pygame.init()
screen     = pygame.display.set_mode((560, 280))
canvas     = pygame.Surface((280, 280))
canvas.fill((0, 0, 0))
pygame.display.set_caption("Draw a digit")
font_big   = pygame.font.Font(None, 120)
font_small = pygame.font.Font(None, 36)
clock      = pygame.time.Clock()

digits, confs = [0, 0, 0], [0.0, 0.0, 0.0]
drawing       = False

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 3:
            canvas.fill((0, 0, 0))   
            digits, confs = [0,0,0], [0.0,0.0,0.0]
        if event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            digits, confs = predict(canvas)  

    if pygame.mouse.get_pressed()[0]:
        mx, my = pygame.mouse.get_pos()
        if mx < 280:
            pygame.draw.circle(canvas, (255,255,255), (mx, my), 8)

    
    screen.fill((20, 20, 20))
    screen.blit(canvas, (0, 0))
    pygame.draw.rect(screen, (40, 40, 40), (70, 50, 140, 180), 1)
    pred_text = font_big.render(str(digits[0]), True, (255, 255, 255))
    screen.blit(pred_text, (340, 60))
    conf_text = font_small.render(f"{confs[0]*100:.1f}%", True, (100, 255, 100))
    screen.blit(conf_text, (310, 180))
    for i in range(1, 3):
        t = font_small.render(f"{digits[i]}: {confs[i]*100:.1f}%", True, (150,150,150))
        screen.blit(t, (290, 210 + i*30))

    hint = font_small.render("draw | right click = clear", True, (80,80,80))
    screen.blit(hint, (10, 255))
    pygame.display.flip()
    clock.tick(60)

pygame.quit()