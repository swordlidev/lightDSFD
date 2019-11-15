#encoding:utf-8
'''
txt描述文件 image_name.jpg num x y w h 1 x y w h 1 这样就是说一张图片中有两个人脸
'''
import os
import sys
import os.path

import random
import math
import numpy as np

import torch
import torch.utils.data as data
import torchvision.transforms as transforms

import cv2

from encoder import DataEncoder

class ListDataset(data.Dataset):

	def __init__(self,list_file,root=None,train=True,
			transform=None,image_size=96,
			small_threshold=5,big_threshold=60,
			setmin=6,setmax=50,
			fm_size=None,ac_size=None,ac_density=None,
			stride=4,offset=12):
		print('data init')
		self.image_size=image_size
		self.root=root
		self.train = train
		self.transform=transform
		self.fnames = []
		self.boxes = []
		self.labels = []
		self.small_threshold = float(small_threshold)#img_48:8,45,10,40  img_36:8,36,10,35
		self.big_threshold = float(big_threshold)
		self.data_encoder = DataEncoder(img_size=image_size,
			fm_size=fm_size,ac_size=ac_size,ac_density=ac_density,
			stride=stride,offset=offset)
		self.setmin = setmin
		self.setmax = setmax

		with open(list_file) as f:
			lines  = f.readlines()

		for line in lines:
			splited = line.strip().split()
			self.fnames.append(splited[0])
			num_faces = int(splited[1])
			box=[]
			label=[]
			for i in range(num_faces):
				x = float(splited[2+5*i])
				y = float(splited[3+5*i])
				w = float(splited[4+5*i])
				h = float(splited[5+5*i])
				c = int(splited[6+5*i])
				box.append([x,y,x+w,y+h])
				label.append(c)
			self.boxes.append(torch.Tensor(box))
			self.labels.append(torch.LongTensor(label))
		self.num_samples = len(self.boxes)

	def __getitem__(self,idx):
		while True:
			fname = self.fnames[idx]
			img = cv2.imread(os.path.join(self.root+fname))
			if img is None:
				idx = random.randrange(0,self.num_samples)
				continue
			imh, imw, _ = img.shape
			boxes = self.boxes[idx].clone()
			labels = self.labels[idx].clone()
			boxwh = boxes[:,2:] - boxes[:,:2]
			center = (boxes[:,:2] + boxes[:,2:]) / 2.
#			boxar = boxwh[:,0] * boxwh[:,1]
			ratio = boxwh.max(1)[0]/boxwh.min(1)[0]
			mask = (boxwh[:,0]>=self.setmin) & (boxwh[:,1]>=self.setmin) & (ratio<float(self.setmax)/self.setmin) & (center[:,0]>0) & (center[:,0]<imw-1) & (center[:,1]>0) & (center[:,1]<imh-1)
			if mask.any():
				break
			else:
				idx = random.randrange(0,self.num_samples)	
		if self.train:
			while True:
				bbox_idx = random.randint(0,boxwh.size(0)-1)
#				area = boxwh[bbox_idx][0]*boxwh[bbox_idx][1]
#				if area >= self.setmin**2:
				if mask[bbox_idx]:
					break
#			if area > self.setmax**2:
			if max(boxwh[bbox_idx][0], boxwh[bbox_idx][1]) > self.setmax:
				oh,ow,_ = img.shape
				fct_min = self.setmin / min(boxwh[bbox_idx][0], boxwh[bbox_idx][1])
				fct_max = self.setmax / max(boxwh[bbox_idx][0], boxwh[bbox_idx][1])
#				tgt_size = random.randint(self.setmin, self.setmax)
#				factor = tgt_size / math.sqrt(area)
#				factor = tgt_size / max(boxwh[bbox_idx][0], boxwh[bbox_idx][1])
				factor = random.uniform(fct_min, fct_max)
				img = cv2.resize(img, (0,0), fx=factor, fy=factor)
				h,w,_ = img.shape
				boxes *= torch.Tensor([float(w)/ow,float(h)/oh,float(w)/ow,float(h)/oh]).expand_as(boxes)
				new_center = (boxes[:,:2] + boxes[:,2:]) / 2
				tmp = (new_center[:,0]>0) & (new_center[:,0]<w) & (new_center[:,1]>0) & (new_center[:,1]<h)
				if not tmp.any():
					print 'center:', center
					print imw, imh
					print 'new_center:', new_center
					print w,h
				assert tmp.any()

			else:
				h,w,_ = img.shape
				center = (boxes[:,:2] + boxes[:,2:]) / 2
				tmp = (center[:,0]>0) & (center[:,0]<w-1) & (center[:,1]>0) & (center[:,1]<h-1)
				if not tmp.any():
					print 'center:', center
					print w,h
				assert tmp.any()

			boxwh = boxes[:,2:]-boxes[:,:2]
			new_mask = (boxwh[:,0] > self.small_threshold) & (boxwh[:,1] > self.small_threshold) & (boxwh[:,0] < self.big_threshold) & (boxwh[:,1] < self.big_threshold)
			if not new_mask.any():
				print boxes
			assert new_mask.any()

			if max(h,w) < self.image_size:
				img, boxes, labels = self.supple_filter(img, boxes, labels)
			elif h >= self.image_size and w >= self.image_size:
				img, boxes, labels = self.random_crop(img, boxes, labels, bbox_idx)
			else:
				img, boxes, labels = self.supple(img, boxes, labels)
				img, boxes, labels = self.random_crop(img, boxes, labels, bbox_idx)
			if random.random() < 0.5:
				img = self.random_bright(img)
				img = self.random_contrast(img)
				img = self.random_saturation(img)
				img = self.random_hue(img)
			else:
				img = self.random_bright(img)
				img = self.random_saturation(img)
				img = self.random_hue(img)
				img = self.random_contrast(img)
			img, boxes = self.random_flip(img, boxes)
			boxwh = boxes[:,2:] - boxes[:,:2]
			# print('boxwh', boxwh)
			
		h,w,_ = img.shape
		assert (h==w and h==self.image_size)
#		img = cv2.resize(img,(self.image_size,self.image_size))
	
		boxes_wh = boxes[:,2:]-boxes[:,:2]
		if ((boxes_wh[:,0]==0) | (boxes_wh[:,1]==0)).any():
			print boxes

#		save_path = '/home/michael/data/tmp/wider_acn/'
#		cv2.imwrite(save_path+'%d_old.jpg'%idx, img)
#		self.visual(img, boxes, idx)
#		cv2.imwrite(save_path+'%d_new.jpg'%idx, img)
#		print 'idx:', idx
#		print 'boxes:', boxes
#		print 'label:', labels
		boxes /= torch.Tensor([w,h,w,h]).expand_as(boxes)
		for t in self.transform:
			img = t(img)
		loc_target,conf_target = self.data_encoder.encode(idx,boxes,labels)

		return img,loc_target,conf_target

	def random_getim(self):
		idx = random.randrange(0,self.num_samples)
		fname = self.fnames[idx]
		img = cv2.imread(os.path.join(self.root+fname))
		boxes = self.boxes[idx].clone()
		labels = self.labels[idx]
		
		return img, boxes, labels

	def __len__(self):
		return self.num_samples
	
	def random_flip(self, im, boxes):
		if random.random() < 0.5:
			im_lr = np.fliplr(im).copy()
			h,w,_ = im.shape
			xmin = w - boxes[:,2]
			xmax = w - boxes[:,0]
			boxes[:,0] = xmin
			boxes[:,2] = xmax
			return im_lr, boxes
		return im, boxes

	def visual(self, im, boxes, idx):
		save_path = '/home/michael/data/tmp/wider_acn/%d.jpg'%idx
		for j, (box) in enumerate(boxes):
			x1 = int(box[0])
			x2 = int(box[2])
			y1 = int(box[1])
			y2 = int(box[3])
			cv2.rectangle(im, (x1, y1+2), (x2, y2), (0,255,0),2)
		cv2.imwrite(save_path, im)

	def supple(self, im, boxes, labels):
		h,w,_ = im.shape
		im = cv2.copyMakeBorder(im, 0, max(0, self.image_size-h), 0, max(0, self.image_size-w), cv2.BORDER_CONSTANT, value=0)
		return im, boxes, labels
		

	def supple_filter(self, im, boxes, labels):
		h,w,_ = im.shape
		im = cv2.copyMakeBorder(im, 0, max(0, self.image_size-h), 0, max(0, self.image_size-w), cv2.BORDER_CONSTANT, value=0)
		boxwh = boxes[:,2:] - boxes[:,:2]
		mask = (boxwh[:,0] > self.small_threshold) & (boxwh[:,1] > self.small_threshold) & (boxwh[:,0] < self.big_threshold) & (boxwh[:,1] < self.big_threshold)
		if not mask.any():
			print boxes
		assert mask.any()
		selected_boxes = boxes.index_select(0, mask.nonzero().squeeze(1))
		selected_labels = labels.index_select(0, mask.nonzero().squeeze(1))
		return im, selected_boxes, selected_labels

	def random_crop(self, im, boxes, labels, bbox_idx):
		imh, imw, _ = im.shape
		w = self.image_size
		h = w
		tgt_box = boxes[bbox_idx]
		#print 'tgt:', tgt_box
		if tgt_box[0] <= 0 or imw == w:
			x = 0
		elif tgt_box[2] >= imw:
			x = imw-1-w
		else:
			x_min = int(max(0, tgt_box[2]-w))
			x_max = int(min(tgt_box[0], imw-w))
			x = random.randint(x_min, x_max)
		if tgt_box[1] <= 0 or imh == h:
			y = 0
		elif tgt_box[3] >= imh:
			y = imh-1-h
		else:
			y_min = int(max(0, tgt_box[3]-h))
			y_max = int(min(tgt_box[1], imh-h))
			y = random.randint(y_min, y_max)
		#print 'xy:', x, y
		roi = torch.Tensor([[x, y, x+w, y+h]])
		center = (boxes[:,:2] + boxes[:,2:]) / 2
		roi2 = roi.expand(len(center), 4)
		mask = (center > roi2[:,:2]) & (center < roi2[:,2:]+1)
		mask = mask[:,0] & mask[:,1]
		if not mask.any():
			print 'roi:', roi
			print 'center:', center
			print 'box:', boxes
			print 'img:', imw, imh
		assert mask.any()

		selected_boxes = boxes.index_select(0, mask.nonzero().squeeze(1))
		img = im[y:y+h, x:x+w, :]
		tmph, tmpw, _ = img.shape
		if tmph != tmpw:
			print tgt_box[0], tgt_box[2], x, y, imw, imh, tmph, tmpw
		assert tmph==tmpw
		selected_boxes[:,0].add_(-x)#.clamp_(min=0, max=w)
		selected_boxes[:,1].add_(-y)#.clamp_(min=0, max=h)
		selected_boxes[:,2].add_(-x)#.clamp_(min=0, max=w)
		selected_boxes[:,3].add_(-y)#.clamp_(min=0, max=h)
		#print selected_boxes
		boxwh = selected_boxes[:,2:] - selected_boxes[:,:2]
		mask = (boxwh[:,0] > self.small_threshold) & (boxwh[:,1] > self.small_threshold) & (boxwh[:,0] < self.big_threshold) & (boxwh[:,1] < self.big_threshold)
		if not mask.any():
			print selected_boxes
			print 'boxes:', boxes
			print 'roi:', roi
			print 'center:', center
			print 'idx:', bbox_idx
			print 'img:', imw, imh
			cv2.imwrite('wrong.jpg', img)
		assert mask.any()

		selected_boxes_selected = selected_boxes.index_select(0, mask.nonzero().squeeze(1))

		selected_labels = labels.index_select(0, mask.nonzero().squeeze(1))

		return img, selected_boxes_selected, selected_labels

	def random_bright(self, im, delta=32):
		if random.random() > 0.5:
			im = im + random.randrange(-delta,delta)
			im = im.clip(min=0,max=255).astype(np.uint8)
		return im

	def random_contrast(self, im):
		if random.random() > 0.5:
			alpha = random.uniform(0.5, 1.5) 
			im = im * alpha
			im = im.clip(min=0,max=255).astype(np.uint8)
		return im

	def random_saturation(self, im):
		if random.random() > 0.5:
			alpha = random.uniform(0.5, 1.5)
			hsv_im = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
			hsv_im = hsv_im * [1.0, alpha, 1.0]
			hsv_im = hsv_im.clip(min=0, max=255).astype(np.uint8)
			im = cv2.cvtColor(hsv_im, cv2.COLOR_HSV2BGR)
		return im

	def random_hue(self, im, delta=18):
		if random.random() > 0.5:
			alpha = random.randrange(-delta,delta)
			hsv_im = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
			hsv_im = hsv_im + [alpha, 0, 0]
			hsv_im = hsv_im.clip(min=0, max=179).astype(np.uint8)
			im = cv2.cvtColor(hsv_im, cv2.COLOR_HSV2BGR)
		return im

	def testGet(self, idx):
		fname = self.fnames[idx]
		img = cv2.imread(os.path.join(self.root,fname))
		cv2.imwrite('test_encoder_source.jpg', img)
		boxes = self.boxes[idx].clone()
		# print(boxes)
		labels = self.labels[idx].clone()

		for box in boxes:
			cv2.rectangle(img, (int(box[0]),int(box[1])), (int(box[2]),int(box[3])), (0,0,255))
		cv2.imwrite(fname, img)

		if self.train:
			img, boxes, labels = self.random_crop(img, boxes, labels)
			img = self.random_bright(img)
			img, boxes = self.random_flip(img, boxes)

		h,w,_ = img.shape
		boxes /= torch.Tensor([w,h,w,h]).expand_as(boxes)

		img = cv2.resize(img,(self.image_size,self.image_size))
		for t in self.transform:
			img = t(img)

		print(idx, fname, boxes)

		return img, boxes, labels

if __name__ == '__main__':
	file_root = '/home/lxg/codedata/aflw/'
	train_dataset = ListDataset(root=file_root,list_file='box_label.txt',train=True,transform = [transforms.ToTensor()] )
	print('the dataset has %d image' % (len(train_dataset)))
	for i in range(len(train_dataset)):
		print(i)
		item = random.randrange(0, len(train_dataset))
		item = item
		img, boxes, labels = train_dataset.testGet(item)
		# img, boxes = train_dataset[item]
		img = img.numpy().transpose(1,2,0).copy()*255
		train_dataset.data_encoder.test_encode(boxes, img, labels)

		boxes = boxes.numpy().tolist()
		w,h,_ = img.shape
		# print('img', img.shape)
		# print('boxes', boxes.shape)

		for box in boxes:
			x1 = int(box[0]*w)
			y1 = int(box[1]*h)
			x2 = int(box[2]*w)
			y2 = int(box[3]*h)
			cv2.rectangle(img, (x1,y1), (x2,y2), (0,0,255))
			boxw = x2-x1
			boxh = y2-y1
			print(boxw,boxh, box)
			if boxw is 0 or boxh is 0:
				raise 'zero width'
			
		cv2.imwrite('test'+str(i)+'.jpg', img)
		if i == 0:
			break
