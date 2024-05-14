#Importing necessary libraries
from tkinter import *
from PIL import ImageTk, Image  
from tkinter import messagebox
import re
import pickle
import pandas as pd
import joblib
import numpy as np
import tensorflow as tf

# Load the trained model
model_cnnbilstm = tf.keras.models.load_model("Model/cnn_bilstm_model.h5")
# Load the scaler
scaler = joblib.load(open('Extra/scaler.pkl', 'rb'))

#perfrom authentication
def authenticate(un_entry,password_entry1):
    username=un_entry.get()
    password=password_entry1.get()

    if(username=="" or password==""):
        messagebox.showwarning("warning","Please Fill Details")  
    elif(username=="user" and password=="user"):
        show_admin()
    else:
        messagebox.showwarning("warning","Invalid Credentials")  


def login():
    LoginPage = Frame(window)
    LoginPage.grid(row=0, column=0, sticky='nsew')
    LoginPage.tkraise()
    window.title('Cyber Threats Detector')

    #login page
    de1 = Listbox(LoginPage, bg='#2f7a61', width=115, height=50, highlightthickness=0, borderwidth=0)
    de1.place(x=0, y=0)
    de2 = Listbox(LoginPage, bg= '#62bd9f', width=115, height=50, highlightthickness=0, borderwidth=0)
    de2.place(x=606, y=0)

    de3 = Listbox(LoginPage, bg='#8be0c4', width=100, height=33, highlightthickness=0, borderwidth=0)
    de3.place(x=76, y=66)

    de4 = Listbox(LoginPage, bg='#f8f8f8', width=85, height=33, highlightthickness=0, borderwidth=0)
    de4.place(x=606, y=66)
    #  Username
    un_entry = Entry(de4, fg="#333333", font=("yu gothic ui semibold", 12), highlightthickness=2,
                        )
    un_entry.place(x=134, y=170, width=256, height=34)
    un_entry.config(highlightbackground="black", highlightcolor="black")
    un_label = Label(de4, text='• Username', fg="#89898b", bg='#f8f8f8', font=("yu gothic ui", 11, 'bold'))
    un_label.place(x=130, y=140)
    #  Password 
    password_entry1 = Entry(de4, fg="#333333", font=("yu gothic ui semibold", 12), show='*', highlightthickness=2,
                            )
    password_entry1.place(x=134, y=250, width=256, height=34)
    password_entry1.config(highlightbackground="black", highlightcolor="black")
    password_label = Label(de4, text='• Password', fg="#89898b", bg='#f8f8f8', font=("yu gothic ui", 11, 'bold'))
    password_label.place(x=130, y=220)

    # function for show and hide password
    def password_command():
        if password_entry1.cget('show') == '*':
            password_entry1.config(show='')
        else:
            password_entry1.config(show='*')

    # checkbutton 
    checkButton = Checkbutton(de4, bg='#f8f8f8', command=password_command, text='show password')
    checkButton.place(x=140, y=288)

  
    #top Login Button
    lob = Label(LoginPage, text='Cyber Attack Detector', font=("yu gothic ui bold", 16), bg='#f8f8f8', fg="#89898b",
                          borderwidth=0, activebackground='#1b87d2')
    lob.place(x=765, y=125)

    lol = Canvas(LoginPage, width=60, height=5, bg='black')
    lol.place(x=826, y=163)

    #  LOGIN  down button 
    loginBtn1 = Button(de4, fg='#f8f8f8', text='Login', bg='#c396e3', font=("yu gothic ui bold", 15),
                       cursor='hand2', activebackground='#c2aed1',command=lambda:authenticate(un_entry,password_entry1))
    loginBtn1.place(x=133, y=340, width=256, height=50)
    #User icon 
    u_icon = Image.open('pics\\user.png')
    photo = ImageTk.PhotoImage(u_icon)
    Uicon_label = Label(de4, image=photo, bg='#f8f8f8')
    Uicon_label.image = photo
    Uicon_label.place(x=103, y=173)

    #  password icon 
    password_icon = Image.open('pics\\key.png')
    photo = ImageTk.PhotoImage(password_icon)
    password_icon_label = Label(de4, image=photo, bg='#f8f8f8')
    password_icon_label.image = photo
    password_icon_label.place(x=103, y=253)


    #  Left Side Picture 
    side_image = Image.open('pics\\h1.png')
    side_image = side_image.resize((400,400))
    photo = ImageTk.PhotoImage(side_image)
    side_image_label = Label(de3, image=photo, bg='#ffdb99')
    side_image_label.image = photo
    side_image_label.place(x=70, y=65)



def get_prediction():

    alltext=input_text.get("1.0",'end')
    if alltext=='' or alltext=='\n':
        messagebox.showinfo("Alert","Fill the empty field")  
    else:
        list1=alltext.split(",")
        print(list1)
        values=[float(x)for x in list1]
        print(values)


        column_names = ['Flow_Duration', 'Tot_Fwd_Pkts', 'Tot_Bwd_Pkts', 'TotLen_Fwd_Pkts', 'TotLen_Bwd_Pkts',
                    'Fwd_Pkt_Len_Max', 'Fwd_Pkt_Len_Min', 'Fwd_Pkt_Len_Mean', 'Fwd_Pkt_Len_Std', 'Bwd_Pkt_Len_Max',
                    'Bwd_Pkt_Len_Min', 'Bwd_Pkt_Len_Mean', 'Bwd_Pkt_Len_Std', 'Flow_Byts/s', 'Flow_Pkts/s',
                    'Flow_IAT_Mean', 'Flow_IAT_Std', 'Flow_IAT_Max', 'Flow_IAT_Min', 'Fwd_IAT_Tot', 'Fwd_IAT_Mean',
                    'Fwd_IAT_Std', 'Fwd_IAT_Max', 'Fwd_IAT_Min', 'Bwd_IAT_Tot', 'Bwd_IAT_Mean', 'Bwd_IAT_Std',
                    'Bwd_IAT_Max', 'Bwd_IAT_Min', 'Bwd_PSH_Flags', 'Fwd_Header_Len', 'Bwd_Header_Len', 'Fwd_Pkts/s',
                    'Bwd_Pkts/s', 'Pkt_Len_Min', 'Pkt_Len_Max', 'Pkt_Len_Mean', 'Pkt_Len_Std', 'Pkt_Len_Var',
                    'SYN_Flag_Cnt', 'PSH_Flag_Cnt', 'ACK_Flag_Cnt', 'ECE_Flag_Cnt', 'Down/Up_Ratio', 'Pkt_Size_Avg',
                    'Fwd_Seg_Size_Avg', 'Bwd_Seg_Size_Avg', 'Subflow_Fwd_Pkts', 'Subflow_Fwd_Byts', 'Subflow_Bwd_Pkts',
                    'Subflow_Bwd_Byts', 'Init_Bwd_Win_Byts', 'Fwd_Act_Data_Pkts', 'Active_Mean', 'Active_Max',
                    'Active_Min', 'Idle_Mean', 'Idle_Std', 'Idle_Max', 'Idle_Min']

        # Create a DataFrame from the lists
        input_data = pd.DataFrame([values], columns=column_names)

        # Perform scaling
        scaled_input = scaler.transform(input_data)

        scaled_input_reshaped = scaled_input.reshape((scaled_input.shape[0], scaled_input.shape[1], 1))
        # Perform predictions
        predictions = model_cnnbilstm.predict(scaled_input_reshaped)
        print(predictions)

        # Get the predicted class label
        predicted_class = np.argmax(predictions)

        # Mapping predicted class to attack type
        attack_types = {
            0: 'Normal',
            1: 'Mirai',
            2: 'Scan',
            3: 'DoS',
            4: 'MITM ARP Spoofing'
        }

        predicted_attack = attack_types[predicted_class]

        print("Predicted Attack Type:", predicted_attack)

        messagebox.showinfo("Prediction",predicted_attack)


def show_admin():
    Admin=Frame(window,bg="#5845d3")
    Admin.grid(row=0, column=0, sticky='nsew')
    Admin.tkraise()
    window.title('Cyber Threats Detector')


    de2 = Listbox(Admin, bg='#d2cad9', width=200, height=42, highlightthickness=0, borderwidth=0)
    de2.place(x=0, y=0)

    input_label = Label(de2, text='Perform Prediction', font=('Arial', 24, 'bold'), bg='#d2cad9')
    input_label.place(x=485, y=38)
    i1 = Canvas(de2, width=164, height=2, bg='#333333',highlightthickness=0)
    i1.place(x=530, y=82)

    global input_text
    input_text = Text(de2, font=('Arial', 18), bd=2, width=40, height=6)
    input_text.place(x=370, y=200)

  
    # Buttons
    p_b_image=Image.open('pics\\pre_button.png')
    p_b_photo=ImageTk.PhotoImage(p_b_image)
    predict_Btn1 = Button(de2, image=p_b_photo, bg='#d2cad9',
                       cursor='hand2',bd=0, activebackground='#6699ff',command=lambda:get_prediction())
    predict_Btn1.image=p_b_photo
    predict_Btn1.place(x=523, y=420)


    refresh_image=Image.open('pics\\re_button.png')
    refresh_photo=ImageTk.PhotoImage(refresh_image)
    refresh_Btn1 = Button(de2, image=refresh_photo, bg='#d2cad9',
                       cursor='hand2',bd=0, activebackground='#00cc66',command=lambda:show_admin())
    refresh_Btn1.image=refresh_photo
    refresh_Btn1.place(x=700, y=420)

    
window = Tk()
window.rowconfigure(0, weight=1)
window.columnconfigure(0, weight=1)
window.geometry("1200x650")
window.maxsize(1200, 650)
window.minsize(1200, 650)
# Window Icon Photo
login()
#show_admin()

window.mainloop()
