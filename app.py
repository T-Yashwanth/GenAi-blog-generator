import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers

## Function To get response from LLAma 2 model
def getLLamaresponse(input_text, no_words, blog_style):
    """
    This function uses the LLama 2 model to generate a blog based on the input parameters.

    Args:
        input_text (str): The topic of the blog.
        no_words (str): The number of words in the blog.
        blog_style (str): The style of the blog (e.g. researcher, data scientist, etc.).

    Returns:
        str: The generated blog.
    """
    ### LLama2 model
    llm = CTransformers(model='models/llama-2-7b-chat.Q8_0.gguf',
                        model_type='llama',
                        config={'max_new_tokens':256,
                                'temperature':0.01})
    
    ## Prompt Template
    template = """
        Write a blog for {blog_style} job profile for a topic {input_text}
        within {no_words} words.
    """
    
    prompt = PromptTemplate(input_variables=["blog_style","input_text",'no_words'],
                            template=template)
    
    ## Generate the response from the LLama 2 model
    response = llm(prompt.format(blog_style=blog_style,input_text=input_text,no_words=no_words))
    return response

# Streamlit UI
st.set_page_config(page_title="Generate Blogs",
                    layout='centered',
                    initial_sidebar_state='collapsed')

st.header("Generate Blogs")

input_text = st.text_input("Enter the Blog Topic")

## Creating two more columns for additional 2 fields
col1, col2 = st.columns([5, 5])

with col1:
    no_words = st.text_input('No of Words')

with col2:
    blog_style = st.selectbox('Writing the blog for',
                            ('Researchers','Data Scientist','Common People'), index=0)

submit = st.button("Generate")

## Final response
if submit:
    response = getLLamaresponse(input_text, no_words, blog_style)
    st.write(response)
