# from distutils.command.upload import upload
from tkinter.tix import IMAGE
from django.contrib import messages
from django.shortcuts import  render, redirect

from xml.etree.ElementTree import tostring
from .forms import NewUserForm
from django.contrib.auth.forms import AuthenticationForm #add this
from django.contrib.auth import login, logout, authenticate #add this
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User
import cv2
import numpy as np
from django.conf import settings
from django.core.files.storage import FileSystemStorage
from django.shortcuts import render
from tensorflow.keras.applications import vgg16
from tensorflow.keras.models import load_model
import base64
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image , ImageTk 
from tensorflow.keras.optimizers import Adam

#from keras.optimizers import Adam
from tensorflow.keras.applications.imagenet_utils import decode_predictions
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.python.keras.backend import set_session
global up
up=""

def index(request):
	return render(request, 'index.html')

# def classification(request):
# 	return render(request, 'classification.html')

def classification1(request):
	return render(request, 'classification1.html')

def classification1(request):
    if request.method == 'POST' and request.FILES['upload']:
        upload = request.FILES['upload']
        up=upload
        fn = up
        print("uploaded:",up)
        
    
        fss = FileSystemStorage()
        file = fss.save(upload.name, upload)
        print("save ", file)
        file_url = fss.url(file)
        print("url:",file_url)
        
       
        imgpath = up
        
        fn = up
        IMAGE_SIZE = 64
        LEARN_RATE = 1.0e-4
        CH=3
        print(fn)
        if fn!="":

            #img = cv2.imread('C:/new/21C9588-Rice prediction/rice_web/rice_web/media/image.png',0)
            img = Image.open(fn)
            img = np.array(img.convert('L'))
            
            
            print(img)
            
            filename1 = 'media/grey.jpeg'
            cv2.imwrite(filename1, img)
            
        
            file_url1 = fss.url(filename1)
            print("url:",file_url1) 
             
        
            #convert into binary
            ret,binary = cv2.threshold(img,160,255,cv2.THRESH_BINARY)# 160 - threshold, 255 - value to assign, THRESH_BINARY_INV - Inverse binary
            #img.save('media/binary.jpeg')
            filename2 = 'media/binary.jpeg'
            
            cv2.imwrite(filename2, binary)
            # Model Architecture and Compilation
        
            model = load_model(r'D:\Skin__disease_web\Skin__disease_web\rice_web\skin_model.h5')
                
            # adam = Adam(lr=LEARN_RATE, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0)
            # model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
            
            img = Image.open(fn)
            img = img.resize((IMAGE_SIZE,IMAGE_SIZE))
            img = np.array(img)
            
            img = img.reshape(1,IMAGE_SIZE,IMAGE_SIZE,3)
            
            img = img.astype('float32')
            img = img / 255.0
            print('img shape:',img)
            prediction = model.predict(img)
            print(np.argmax(prediction))
            disease=np.argmax(prediction)
            print(disease)
            if disease == 0:
                Cd="Chickenpox"
                ans="Precaution- Avoid picking or squeezing the blisters. Limit makeup use to avoid irritation and spread of infection"

            elif disease == 1:
                Cd="Nail-Fungus"
                ans="Precaution-  Do not scrub or dry the skin too hard or for too long. This helps to avoid further irritation or infection"
              
            elif disease == 2:
                Cd="Impetigo"
                ans="Precaution-  Once diagnosed with impetigo, a doctor will recommend a treatment plan. It's important to follow the plan based on the severity of symptoms, age, sex, health, and lifestyle factors"
                
            elif disease == 3:
                Cd= "Shingles"
                ans="Precaution- To reduce your risk of melanoma and other types of skin cancer, avoid tanning lamps and beds. It's also advised to manage stress, which can trigger shingles."
                
            elif disease == 4:
                Cd= "cutaneous-larva-migrans"
                ans="Precaution- After exposure, dry your skin well, apply an antifungal foot powder, and moisturize your nails to avoid infection"
               
            
           
                
                
            A=Cd
        
            
            
          
            return render(request, "fruits.html", {"predictions1": A,"ans1":ans,'file_url': file_url})  

    else:
    
        return render(request, "classification1.html") #
    #return render(request, "classification1.html")
     
def index(request):        
        return render(request, "index.html")

def fruits(request):        
        return render(request, "fruits.html")

def register(request):
	if request.method == "POST":
		form = NewUserForm(request.POST)
		if form.is_valid():
			user = form.save()
			login(request, user)
			messages.success(request, "Registration successful." )
			return redirect('/login1/')
		messages.error(request, "Unsuccessful registration. Invalid information.")
	form = NewUserForm()
	return render (request=request, template_name="register.html", context={"register_form":form})

def about(request):
      return  render(request,"about.html")


def login1(request):
	if request.method == "POST":
		form = AuthenticationForm(request, data=request.POST)
		if form.is_valid():
			username = form.cleaned_data.get('username')
			password = form.cleaned_data.get('password')
			user = authenticate(username=username, password=password)
			if user is not None:
				login(request, user)
				messages.info(request, f"You are now logged in as {username}.")
				return redirect('index')
			else:
				messages.error(request,"Invalid username or password.")
		else:
			messages.error(request,"Invalid username or password.")
	form = AuthenticationForm()
	return render(request=request, template_name="login.html", context={"login_form":form})

def logout_request(request):
	logout(request)
	messages.info(request, "You have successfully logged out.") 
	return redirect('login1')
