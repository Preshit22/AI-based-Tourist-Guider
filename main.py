import openai
import pandas as pd
import streamlit as st
import google.generativeai as genai
import os
from io import BytesIO
import time

from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

openai.api_key = 'your-api-key'
os.environ["GOOGLE_API_KEY"] = 'your-api-key'  

genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
gemini_model = genai.GenerativeModel('gemini-1.0-pro-latest')

chatgpt_prompt = PromptTemplate(
    input_variables=["city", "country", "attraction"],
    template="Give a detailed description of {attraction} in {city}, {country}. The response should only include information about {attraction} in {city}, {country}."
)
gemini_prompt = "Provide an in-depth overview of {attraction} located in {city}, {country}. The response should only include information about {attraction} in {city}, {country}."

chatgpt_model = ChatOpenAI(api_key=openai.api_key, model="gpt-3.5-turbo")

chatgpt_chain = LLMChain(llm=chatgpt_model, prompt=chatgpt_prompt)

def get_gemini_description(country, city, attraction):
    start_time = time.time()
    response = gemini_model.generate_content(gemini_prompt.format(country=country, city=city, attraction=attraction))
    end_time = time.time()
    processing_time = end_time - start_time
    return response.text.strip(), processing_time

def get_attraction_info(country, city, attraction):
    start_time = time.time()
    
    chatgpt_start_time = time.time()
    chatgpt_response = chatgpt_chain.run(city=city, country=country, attraction=attraction).strip()
    chatgpt_end_time = time.time()
    chatgpt_processing_time = chatgpt_end_time - chatgpt_start_time
    
    gemini_response, gemini_processing_time = get_gemini_description(country, city, attraction)
    
    combined_description = f"ChatGPT: {chatgpt_response}\n\nGemini: {gemini_response}"
    
    end_time = time.time()
    total_processing_time = end_time - start_time
    
    return {
        "chatgpt_response": chatgpt_response,
        "gemini_response": gemini_response,
        "combined_description": combined_description,
        "chatgpt_processing_time": chatgpt_processing_time,
        "gemini_processing_time": gemini_processing_time,
        "total_processing_time": total_processing_time
    }

def create_excel(data):
    df = pd.DataFrame(data, columns=["Country", "City", "Attraction Name", "ChatGPT Description", "Gemini Description", "Combined Description", "ChatGPT Processing Time (s)", "Gemini Processing Time (s)", "Total Processing Time (s)"])
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='openpyxl')
    df.to_excel(writer, index=False, sheet_name='Tourist Attractions')
    writer.close()
    output.seek(0)  
    return output


st.title('AI-based-Tourist-Guider')
country = st.text_input('Country')
city = st.text_input('City')
attraction_name = st.text_input('Attraction Name')

if st.button('Get Information'):
    if country and city and attraction_name:
        info = get_attraction_info(country, city, attraction_name)
        
        data = [[
            country, city, attraction_name, 
            info['chatgpt_response'], info['gemini_response'], info['combined_description'], 
            info['chatgpt_processing_time'], info['gemini_processing_time'], info['total_processing_time']
        ]]
        
        excel_data = create_excel(data)
        st.success('Excel file created successfully!')
        st.download_button(label='Download Excel', data=excel_data, file_name='tourist_attractions.xlsx', mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
        
        st.subheader("Results")
        st.write(f"**Country:** {country}")
        st.write(f"**City:** {city}")
        st.write(f"**Attraction Name:** {attraction_name}")
        st.write("**ChatGPT Description:**")
        st.write(info['chatgpt_response'] if info['chatgpt_response'] else "No description available.")
        st.write("**Gemini Description:**")
        st.write(info['gemini_response'] if info['gemini_response'] else "No description available.")
        st.write("**Combined Description:**")
        st.write(info['combined_description'] if info['combined_description'] else "No combined description available.")
        
        st.write(f"**ChatGPT Processing Time:** {info['chatgpt_processing_time']:.2f} seconds")
        st.write(f"**Gemini Processing Time:** {info['gemini_processing_time']:.2f} seconds")
        st.write(f"**Total Processing Time:** {info['total_processing_time']:.2f} seconds")
    else:
        st.error("Please provide all inputs: Country, City, and Attraction Name.")
