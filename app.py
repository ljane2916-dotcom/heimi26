import streamlit as st
from transformers import pipeline

# Function Part

def img2text(url):
    image_to_text_model = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
    text = image_to_text_model(url)[0]["generated_text"]
    return text

def text2story(scenario):
    refined_prompt = f"A very short, sweet, and happy bedtime story for a 5-year-old kid: {scenario}"
    story_pipe = pipeline("text-generation", model="pranavpsv/genre-story-generator-v2")
    story_results = story_pipe(
        refined_prompt,
        max_new_tokens=80,
        do_sample=True,
        temperature=0.7,
        truncation=True
    )
    full_text = story_results[0]['generated_text']
    # Ensure the return statement is aligned with the code above
    story = full_text.replace(refined_prompt, "").strip()
    return story
    
def text2audio(story_text):
    audio_pipe = pipeline("text-to-audio", model="Matthijs/mms-tts-eng")
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
