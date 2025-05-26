import tkinter as tk
from tkinter import ttk, LEFT, END

import math
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, RobustScaler
import train_modelCL as TrainM
import time
import warnings 
warnings.filterwarnings("ignore", category=DeprecationWarning)

root = tk.Tk()
root.configure(background="seashell2")
#root.geometry("1300x700")


w, h = root.winfo_screenwidth(), root.winfo_screenheight()
root.geometry("%dx%d+0+0" % (w, h))
root.title("Credit Card Fraud Detection")

##############################################+=============================================================
lbl = tk.Label(root, text="Credit Card Fraud Detection Using Machine Learning", font=('times', 35,' bold '),justify=tk.LEFT, wraplength=1700 ,bg="white",fg="indian red")
lbl.place(x=10, y=5)

frame_CP = tk.LabelFrame(root, text=" Control Panel ", width=200, height=750, bd=5, font=('times', 12, ' bold '),bg="lightblue4",fg="white")
frame_CP.grid(row=0, column=0, sticky='s')
frame_CP.place(x=5, y=60)

frame_display = tk.LabelFrame(root, text=" ---Result--- ", width=1000, height=750, bd=5, font=('times', 12, ' bold '),bg="white",fg="red")
frame_display.grid(row=0, column=0, sticky='s')
frame_display.place(x=210, y=60)

frame_noti = tk.LabelFrame(root, text=" Notification ", width=250, height=750, bd=5, font=('times', 12, ' bold '),bg="lightblue4",fg="white")
frame_noti.grid(row=0, column=0, sticky='nw')
frame_noti.place(x=1330, y=60)

###########################################################################################################
canvas=tk.Canvas(frame_display,bg='#FFFFFF',width=1000,height=600,scrollregion=(0,0,7700,15030))
hbar=tk.Scrollbar(frame_display,orient=tk.HORIZONTAL)
hbar.pack(side=tk.BOTTOM,fill=tk.X)
hbar.config(command=canvas.xview)
vbar=tk.Scrollbar(frame_display,orient=tk.VERTICAL)
vbar.pack(side=tk.RIGHT,fill=tk.Y)
vbar.config(command=canvas.yview)
canvas.config(width=1100,height=680)
canvas.config(xscrollcommand=hbar.set, yscrollcommand=vbar.set)
#canvas.pack(side=LEFT,expand=True,fill=tk.BOTH)
canvas.pack(fill=tk.BOTH, expand=tk.YES)

basepath=r'C:\\Users\\sunil\\Projects\\Credit Card Fraud Detection\\'



def update_label(str_T):
    result_label = tk.Label(frame_noti, text=str_T, font=("italic", 20),justify=tk.LEFT, wraplength=200 ,bg='lightblue4',fg='white' )
    result_label.place(x=10, y=0)


def display_data():

    df=pd.read_csv('C:\\Users\\sunil\\Projects\\Credit Card Fraud Detection\\creditcard.csv')
    df=df.head(500)
    
    
    icol=50
    colsp=350
    canvas.delete("all")
    canvas.create_text(5,15,fill="red",anchor="w",font="Times 16 bold",text="ID")
    canvas.create_line([(90, 0), (90, 50000)], fill='red', tags='grid_line_w')

    canvas.create_text(110,5,fill="red",anchor="nw",font="Times 16 bold",text="Class")
    canvas.create_line([(160, 0), (160, 50000)], fill='red', tags='grid_line_w')
    
    canvas.create_text(210,5,fill="red",anchor="nw",font="Times 16 bold",text="Time")
    #canvas.create_line([(270, 0), (250, 10000)], fill='red', tags='grid_line_w')

    dateL=df.columns[2:-1]
    
    for i in range(len(dateL)):
        
        canvas.create_text(colsp,5,fill="red",anchor="nw",font="Times 16 bold",text=dateL[i])
        colsp=colsp+250

    
    for i in range(len(df)):
        
        canvas.create_text(5,icol,fill="blue",anchor="w",font="Times 15",text=df["ID"][i])
        canvas.create_text(120,icol,fill="blue",anchor="nw",font="Times 15",text=df["Class"][i])
        canvas.create_text(220,icol,fill="blue",anchor="nw",font="Times 15",text=df["Time"][i])
        
        colsp=300
        rowp = 280

        for j in range(len(dateL)):
            canvas.create_text(colsp,icol,fill="blue",anchor="nw",font="Times 15",text=df[dateL[j]][i])
            canvas.create_line([(rowp, 0), (rowp, 50000)], fill='red', tags='grid_line_w')
            colsp=colsp+250
            rowp+=250
        
        
        canvas.create_line([(0, icol-12), (10000, icol-12)], fill='black', tags='grid_line_h')
    
        icol=icol+30

def Process_data():
    
    df=pd.read_csv('C:\\Users\\sunil\\Projects\\Credit Card Fraud Detection\\creditcard.csv')
    rob_scaler = RobustScaler()
    
    df['Amount'] = rob_scaler.fit_transform(df['Amount'].values.reshape(-1,1))
    df['Time'] = rob_scaler.fit_transform(df['Time'].values.reshape(-1,1))
    
    
    
    #----------------------------------------------------------
    df=df.head(500)
    icol=50
    colsp=500
    canvas.delete("all")
    canvas.create_text(5,15,fill="red",anchor="w",font="Times 16 bold",text="ID")
    canvas.create_line([(90, 0), (90, 50000)], fill='red', tags='grid_line_w')

    canvas.create_text(110,5,fill="red",anchor="nw",font="Times 16 bold",text="Class")
    canvas.create_line([(160, 0), (160, 50000)], fill='red', tags='grid_line_w')
    
    canvas.create_text(210,5,fill="red",anchor="nw",font="Times 16 bold",text="Time")
    #canvas.create_line([(270, 0), (250, 10000)], fill='red', tags='grid_line_w')

    dateL=df.columns[2:]
    
    for i in range(len(dateL)):
        
        canvas.create_text(colsp,5,fill="red",anchor="nw",font="Times 16 bold",text=dateL[i])
        colsp=colsp+250

    
    for i in range(len(df)):
        
        canvas.create_text(5,icol,fill="blue",anchor="w",font="Times 15",text=df["ID"][i])
        canvas.create_text(120,icol,fill="blue",anchor="nw",font="Times 15",text=df["Class"][i])
        canvas.create_text(180,icol,fill="blue",anchor="nw",font="Times 15",text=df["Time"][i])
        
        colsp=450
        rowp = 400

        for j in range(len(dateL)):
            canvas.create_text(colsp,icol,fill="blue",anchor="nw",font="Times 15",text=df[dateL[j]][i])
            canvas.create_line([(rowp, 0), (rowp, 50000)], fill='red', tags='grid_line_w')
            colsp=colsp+250
            rowp+=250
        
        
        canvas.create_line([(0, icol-12), (10000, icol-12)], fill='black', tags='grid_line_h')
    
        icol=icol+30
    
        
def train_model(): 
    update_label(f"Model Training Start............... \n")
    
    
    start = time.time()
    StrVal= TrainM.main()
    end = time.time()
    
    ET=f"\n Execution Time: {end-start} seconds \n"
    
    
    msg=f" Model Training Completed.. {ET}" 
    
    
    update_label(msg)

#####################################################################

def result_data():

    Rdf=pd.read_csv('C:\\Users\\sunil\\Projects\\Credit Card Fraud Detection\\Result.csv')
    
    icol=50
    canvas.delete("all")
    canvas.create_text(5,15,fill="red",anchor="w",font="Times 16 bold",text="ID")
    canvas.create_line([(390, 0), (390, 10000)], fill='red', tags='grid_line_w')

    canvas.create_text(400,5,fill="red",anchor="nw",font="Times 16 bold",text="Actual Observation")
    canvas.create_line([(690, 0), (690, 10000)], fill='red', tags='grid_line_w')

    canvas.create_text(700,5,fill="red",anchor="nw",font="Times 16 bold",text="Predicted Observation")

    Rdf=Rdf[['ID','Actual Observations','Predicted Observations']]
    
    for i in range(len(Rdf)):

        canvas.create_text(5,icol,fill="blue",anchor="w",font="Times 15",text=Rdf["ID"][i])
        canvas.create_text(410,icol,fill="blue",anchor="nw",font="Times 15",text=Rdf["Actual Observations"][i])
        canvas.create_text(700,icol,fill="blue",anchor="nw",font="Times 15",text=Rdf["Predicted Observations"][i])
       
        canvas.create_line([(0, icol-12), (10000, icol-12)], fill='black', tags='grid_line_h')
    
        icol=icol+30
        
    accuracy = accuracy_score(Rdf['Actual Observations'], Rdf['Predicted Observations'])
    msg = f"\n\n\n\nThe Accuracy of Model is:\n  {accuracy*100} \n"
    
    update_label(msg)
    
def window():
    root.destroy()


button1 = tk.Button(frame_CP, text=" Load Data ", command=display_data,width=19, height=1, font=('times', 12, ' bold '),bg="white",fg="black")
button1.place(x=5, y=50)

button2 = tk.Button(frame_CP, text=" Data Processing ", command=Process_data,width=19, height=1, font=('times', 12, ' bold '),bg="white",fg="black")
button2.place(x=5, y=150)
#Analysis Electricity Theft data
button3 = tk.Button(frame_CP, text=" Model Training ", command=train_model,width=19, height=1, font=('times', 12, ' bold '),bg="white",fg="black")
button3.place(x=5, y=250)

button4 = tk.Button(frame_CP, text=" Result ", command=result_data,width=19, height=1, font=('times', 12, ' bold '),bg="white",fg="black")
button4.place(x=5, y=350)



exit = tk.Button(frame_CP, text="Exit", command=window, width=19, height=1, font=('times', 12, ' bold '),bg="red",fg="white")
exit.place(x=5, y=550)

root.mainloop()