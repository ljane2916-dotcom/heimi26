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

    # 1. 在 Prompt 里明确字数边界，去掉 "long" 这种词
    prompt = (
        f"Write a short, happy story for a child about {caption}. "
        "The story must be between 60 and 90 words. "
        "Start with 'Once upon a time'. "
        "Use only 5 to 6 sentences. Focus on one simple fun moment."
    )

    # 2. 调整参数：max_new_tokens 是控制长度最有效的武器
    # 1个单词大约 1.3 个 token，100个词大约就是 130 tokens
    result = story_generator(
        prompt,
        max_new_tokens=130,    # 严格上限，防止生成过长
        min_new_tokens=75,     # 严格下限，防止生成过短（确保达到50词）
        do_sample=True,
        temperature=0.7,       # 稍微降低随机性，让逻辑更紧凑
        repetition_penalty=2.0 # 提高惩罚，防止反复说 "the park"
    )

    story = result[0]["generated_text"]
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
