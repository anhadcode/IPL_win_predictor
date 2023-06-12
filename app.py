import streamlit as st
import pickle
import pandas as pd
from PIL import Image

st.image("ipl_logo.png")

st.title("IPL Predictor Probability")

teams= ['Sunrisers Hyderabad',
        'Mumbai Indians',
        'Royal Challengers Bangalore',
        'Kolkata Knight Riders',
        'Kings XI Punjab',
        'Chennai Super Kings',
        'Rajasthan Royals',
        'Delhi Capitals']

cities= ['Hyderabad', 'Bangalore', 'Mumbai', 'Indore', 'Kolkata', 'Delhi',
       'Chandigarh', 'Jaipur', 'Chennai', 'Cape Town', 'Port Elizabeth',
       'Durban', 'Centurion', 'East London', 'Johannesburg', 'Kimberley',
       'Bloemfontein', 'Ahmedabad', 'Cuttack', 'Nagpur', 'Dharamsala',
       'Visakhapatnam', 'Pune', 'Raipur', 'Ranchi', 'Mohali', 'Bengaluru']

logo= {
        'Sunrisers Hyderabad':'srh.png',
        'Mumbai Indians':'mi.png',
        'Royal Challengers Bangalore':'rcb.png',
        'Kolkata Knight Riders':'kkr.png',
        'Kings XI Punjab':'pkings',
        'Chennai Super Kings':"Chennai-Super-Kings.jpg",
        'Rajasthan Royals':'rr.png',
        'Delhi Capitals':'dc.png'
}


#pipe= pickle.load(open('pipe.pkl', 'rb'))

col1, col2= st.columns(2)

with col1:
    battingteam= st.selectbox("Select the batting team", sorted(teams))

with col2:
    bowlingteam= st.selectbox("Select the bowling team", sorted(teams))


city= st.selectbox("Select the city",sorted(cities))
target= st.number_input("Target given by the batting team", step=1)

col3, col4, col5= st.columns(3)

with col3:
    score= st.number_input("current score",  step=1)
with col4:
    overs= st.number_input("overs completed", step=1)
with col5:
    wickets= st.number_input("wickets fallen", step=1)



pipe= pickle.load(open('pipe.pkl','rb'))

if st.button("Predict probablity"):

    runs_left= target- score
    balls_left= 120-(overs*6)
    wickets= 10-wickets
    currentrunrate= score/overs
    requiredrunrate= (runs_left*6)/balls_left

    input_df= pd.DataFrame({'batting_team':[battingteam], 'bowling_team':[bowlingteam], 'city':[city], 'runs_left':[runs_left], 'balls_left':[balls_left],
                            'wickets':[wickets], 'total_runs_x':[target], 'cur_run_rate':[currentrunrate], 'req_run_rate':[requiredrunrate]})

    result= pipe.predict_proba(input_df)
    lossprob= result[0][0]
    winprob= result[0][1]

    col6,col7= st.columns(2)

    with col6:
        image = Image.open(logo[battingteam])
        image_resized = image.resize((250,250))
        st.image(image_resized)
        st.header(battingteam+" - "+str(round(winprob*100))+"%")

    with col7:
        image2 = Image.open(logo[bowlingteam])
        image_resized = image2.resize((250, 250))
        st.image(image_resized)
        st.header(bowlingteam+" - "+str(round(lossprob*100))+"%")

    
    