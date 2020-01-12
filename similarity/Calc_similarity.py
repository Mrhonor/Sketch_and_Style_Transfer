from PIL import Image
import math
import numpy as np
import cv2

def get_cos_distance(list1, list2):

   ss = sum([i*j for i,j in zip(list1, list2)])

   sq1 = math.sqrt(sum([pow(i, 2) for i in list1]))

   sq2 = math.sqrt(sum([pow(i, 2) for i in list2]))

   return float(ss)/(sq1*sq2)

def calc_vector(img):
   two_value_img = img.convert('1')
   x = np.zeros(two_value_img.size[0])
   y = np.zeros(two_value_img.size[1])
   num = 0
   for i in range(two_value_img.size[0]):
      sum = 0
      for j in range(two_value_img.size[1]):
         if(two_value_img.getpixel((i,j)) == 255):
            sum = sum + 1
            num = num + 1
      x[i] = sum

   for i in range(two_value_img.size[1]):
      sum = 0
      for j in range(two_value_img.size[0]):
         if(two_value_img.getpixel((i,j)) == 255):
            sum = sum + 1
      y[i] = sum
   return x/num, y/num

def calc_vector2(img):
   two_value_img = img.convert('1')
   x = np.zeros(two_value_img.size[0])
   num = 0 
   for i in range(two_value_img.size[0]):
      sum = 0
      for j in range(i):
         if(two_value_img.getpixel((j,i-j)) == 255):
            sum = sum + 1
            num = num + 1
      x[i] = sum
   return x/num

def compare_similarity(x1,y1,x2,y2):
   distx = np.linalg.norm(x1 - x2)
   disty = np.linalg.norm(y1 - y2)
   return distx+disty


def pHash(img):
    #加载并调整图片为32x32灰度图片
    img=cv2.resize(img,(64,64),interpolation=cv2.INTER_CUBIC)
    img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img = img.astype(np.float32)
 
   #离散余弦变换
    img = cv2.dct(img)
    img = img[0:8,0:8]
    avg = 0
    hash_str = ''
 
    #计算均值
    for i in range(8):
        for j in range(8):
            avg += img[i,j]
    avg = avg/64
 
    #获得hsah
    for i in range(8):
        for j in range(8):
            if  img[i,j]>avg:
                hash_str=hash_str+'1'
            else:
                hash_str=hash_str+'0'            
    return hash_str
 
def cmpHash(hash1,hash2):
    n=0
    if len(hash1)!=len(hash2):
        return -1
    for i in range(len(hash1)):
        if hash1[i]!=hash2[i]:
            n=n+1
    return n

def find_max_loss(n):
   max = -1
   for i in range(n.size):
      if n[i] > max:
         max = n[i]
         index = i
   return index 

def find_min_loss(n):
   min = 10000
   for i in range(n.size):
      if n[i] < min:
         min = n[i]
         index = i
   return index 

input_path = 'input.jpg'
outcome_path = 'fake/'
IMG_SIZE = 64

PIL_img = Image.open(input_path)

PIL_img = PIL_img.crop((0,0,IMG_SIZE,IMG_SIZE))

img1 = cv2.imread(input_path)
hash1 = pHash(img1[0:64, 0:64])

loss_array = 1000*np.ones(5)
index_array2 = np.zeros(5, dtype=int)

similar_array = np.zeros(15, dtype=float)
index_array = np.zeros(15, dtype=int)

# 直方图方法计算
max_similar = 0

input_hist = PIL_img.histogram()

for i in range(64):
   img = Image.open(outcome_path+'fake_samples_'+str(i)+'.png')
   crop_img = img.crop((0*IMG_SIZE, 0*IMG_SIZE, 1*IMG_SIZE, 1*IMG_SIZE))
   outcome_hist = crop_img.histogram()
   similar = get_cos_distance(input_hist, outcome_hist)
   index = find_min_loss(similar_array)
   similar_array[index] = similar
   index_array[index] = i

for i in range(15):
   print("index:%d, loss:%f"%(index_array[i], similar_array[i]))
   # Image.open(outcome_path+'fake_samples_'+str(int(index_array[i]))+'.png').show()

# print("index:%d, similar:%f"%(max_index, max_similar))
# Image.open(outcome_path+'fake_samples_'+str(max_index)+'.png').show()

# 感知哈希算法
for i in index_array:
   img = cv2.imread(outcome_path+'fake_samples_'+str(i)+'.png')
   crop_img = img[0:64, 0:64]
   hash2 = pHash(crop_img)
   loss = cmpHash(hash1, hash2)
   j = find_max_loss(loss_array)
   loss_array[j] = loss
   index_array2[j] = i   

for i in range(5):
   print("index:%d, loss:%f"%(index_array2[i], loss_array[i]))
   # Image.open(outcome_path+'fake_samples_'+str(int(index_array2[i]))+'.png').show()


# 投影对比法2
x1 = calc_vector2(PIL_img)
min_loss = 110000

for i in index_array2:
   img = Image.open(outcome_path+'fake_samples_'+str(i)+'.png')
   crop_img = img.crop((0*IMG_SIZE, 0*IMG_SIZE, 1*IMG_SIZE, 1*IMG_SIZE))
   x2 = calc_vector2(crop_img)
   loss = np.linalg.norm(x1 - x2)
   print(loss)
   if loss < min_loss:
      min_loss = loss
      index = i   

print("index:%d, loss:%f"%(index, min_loss))
Image.open(outcome_path+'fake_samples_'+str(index)+'.png').show()


# # 投影对比法
# x1, y1 = calc_vector(PIL_img)
# min_loss = 110000

# for i in index_array2:
#    img = Image.open(outcome_path+'fake_samples_'+str(i)+'.png')
#    crop_img = img.crop((0*IMG_SIZE, 0*IMG_SIZE, 1*IMG_SIZE, 1*IMG_SIZE))
#    x2, y2 = calc_vector(crop_img)
#    loss = compare_similarity(x1,y1,x2,y2)
#    print(loss)
#    if loss < min_loss:
#       min_loss = loss
#       index = i   

# print("index:%d, loss:%f"%(index, min_loss))
# Image.open(outcome_path+'fake_samples_'+str(index)+'.png').show()





