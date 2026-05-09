import streamlit as st
from transformers import pipeline

# Function Part

def img2text(url):
    image_to_text_model = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
    text = image_to_text_model(url)[0]["generated_text"]
    return text

def text2story(scenario):
    # 1. 初始化模型
    story_pipe = pipeline("text-generation", model="gpt2")
    
    # 2. 关键改动：给 AI 一个具有引导性的开头，迫使它从图片内容往下写
    # 不要用 "Based on..." 这种总结性的话，要用叙述性的句子
    prompt = f"The {scenario} looked amazing. Suddenly,"
    
    # 3. 运行生成
    story_results = story_pipe(
        prompt,
        max_new_tokens=60,      # 给它足够的空间去发挥
        do_sample=True,
        temperature=0.8,
        top_p=0.9,             # 增加生成质量的采样限制
        truncation=True
    )
    
    # 4. 提取生成的完整文本
    full_text = story_results[0]['generated_text']
    
    # 5. 【修正清理逻辑】：如果 AI 连着你的 Prompt 一起写了，我们保留整体
    # 这样能确保返回的一定是一段完整的、有意义的英文，不会变成 None
    if full_text and len(full_text) > len(prompt):
        return full_text
    else:
        # 如果生成太短，强行让它再试一次或者给一个基于 scenario 的补充
        return f"{scenario} is the start of a great adventure that begins right here."
    
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
