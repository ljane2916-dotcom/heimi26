import streamlit as st
from transformers import pipeline

# Function Part

def img2text(url):
    image_to_text_model = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
    text = image_to_text_model(url)[0]["generated_text"]
    return text

def clean_text(text):
    """
    Clean generated text by removing extra spaces and line breaks.
    """
    text = text.replace("\n", " ")
    text = " ".join(text.split())
    return text.strip()

@st.cache_resource
def load_story_generator():
    return pipeline(
        "text2text-generation", 
        model="MBZUAI/LaMini-Flan-T5-248M"
    )

def text2story(caption):
    story_generator = load_story_generator()
    
    prompt = (
        f"Write a creative and happy short story for a 5-year-old child about {caption}. "
        "The story must start with 'Once upon a time' and be at least 6 sentences long. "
        "Make it a fun adventure with a happy ending."
    )

   
    result = story_generator(
        prompt,
        max_length=160,       
        min_length=80,         
        do_sample=True,        
        temperature=0.8,      
        top_k=50,
        top_p=0.95,
        repetition_penalty=2.5 
    )

    story = result[0]["generated_text"]
    
    if not story.lower().startswith("once upon a time"):
        story = "Once upon a time, " + story

    return clean_text(story)
    
@st.cache_resource
def load_audio_generator():
    return pipeline(
        "text-to-audio",
        model="Matthijs/mms-tts-eng"
    )

def text2audio(story_text):
    audio_pipe = load_audio_generator()
    audio_data = audio_pipe(story_text)
    return audio_data

# Main Part

# Page configuration and header
st.set_page_config(page_title="Teddy's Story Corner", page_icon="🧸")
st.title("🧸 Teddy's Story Corner 🍯")
st.header("Hello! I'm Teddy. Let's make a story together!")

# File uploader for user input
uploaded_file = st.file_uploader("Share your photo with Teddy...")

if uploaded_file is not None:
    # Save the uploaded file locally
    bytes_data = uploaded_file.getvalue()
    with open(uploaded_file.name, "wb") as file:
        file.write(bytes_data)

    st.image(uploaded_file, caption="🎨 Teddy is looking at your picture...", use_column_width=True)

    # Stage 1: Image to Text 
    st.subheader('🐾 Teddy sees...')
    scenario = img2text(uploaded_file.name)
    st.write(f":orange[I think I see {scenario} in this picture!]")

    # Stage 2: Text to Story
    st.subheader('🧸 Teddy is writing a story for you...')
    story = text2story(scenario)
    st.balloons()
    st.write(f":orange[**🐻 Teddy's Honey Tale:**] {story}")

    # Stage 3: Story to Audio
    st.subheader("🎧 Listen to Teddy!")
    audio_output = text2audio(story)

    # Play Button
    if st.button("👂 Click to hear Teddy tell the story!"):
        audio_array = audio_output["audio"]
        sample_rate = audio_output["sampling_rate"]
        st.audio(audio_array, sample_rate=sample_rate)
