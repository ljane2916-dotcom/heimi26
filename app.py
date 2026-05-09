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

    # 1. 在 Prompt 里给出一个明确的“结尾指令”
    # 告诉它必须以特定的方式收尾，这样它会更有目的地书写
    prompt = (
        f"Write a short children's story about {caption}. "
        "Structure: \n"
        "1. Start with 'Once upon a time'. \n"
        "2. Describe a happy moment. \n"
        "3. END the story with a final happy sentence like 'They all went home with a big smile.' \n"
        "Limit the story to 5-7 sentences and about 80 words."
    )

    # 2. 关键参数微调
    result = story_generator(
        prompt,
        max_new_tokens=150,    # 稍微调大一点，给结尾留出“呼吸空间”，防止被切断
        min_new_tokens=80,     # 确保达到 50 词以上
        do_sample=True,
        temperature=0.8, 
        repetition_penalty=3.5, # 【重要】显著提高惩罚，防止它一直念叨 "park" 和 "play"
        no_repeat_ngram_size=3  # 防止连续三个词重复
    )

    story = result[0]["generated_text"]
    
    # 3. 兜底逻辑：如果模型还是没写完，我们手动检查一下标点
    # 如果最后一句没有句号，说明没写完
    story = story.strip()
    if not story.endswith(('.', '!', '?')):
        # 尝试补一个温暖的结尾词
        story += " and they lived happily ever after."

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
