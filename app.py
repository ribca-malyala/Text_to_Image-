
"""
Text-to-Image Generator
Generates images from text prompts using Hugging Face's Inference API
(FLUX.1-schnell / Stable Diffusion XL). Needs a FREE Hugging Face token.
"""

import os
from io import BytesIO

import streamlit as st
from dotenv import load_dotenv
from huggingface_hub import InferenceClient


load_dotenv()

st.set_page_config(
    page_title="Text to Image",
    page_icon="🌷",
    layout="centered"
)


# ---------------- Internal CSS ----------------
st.markdown("""
<style>

/* Main background */
.stApp {
    background:
        radial-gradient(circle at 10% 10%, #ffe5ef 0%, transparent 25%),
        radial-gradient(circle at 90% 20%, #eee2ff 0%, transparent 25%),
        radial-gradient(circle at 50% 100%, #fff0d9 0%, transparent 30%),
        #fffafd;
}

/* Main container */
.main .block-container {
    max-width: 850px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

/* Tulip decoration */
.tulip-decoration {
    text-align: center;
    font-size: 30px;
    letter-spacing: 10px;
    margin-bottom: 5px;
    animation: floatFlowers 3s ease-in-out infinite;
}

@keyframes floatFlowers {
    0% {
        transform: translateY(0px);
    }

    50% {
        transform: translateY(-5px);
    }

    100% {
        transform: translateY(0px);
    }
}

/* Main title */
h1 {
    text-align: center;
    color: #b64f78 !important;
    font-family: Georgia, serif;
    font-size: 42px !important;
    margin-bottom: 5px;
}

/* Subtitle */
.stApp p {
    color: #6d5661;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(
        180deg,
        #fff0f6 0%,
        #fff8fb 45%,
        #f5efff 100%
    );

    border-right: 2px solid #f0c5d7;
}

/* Sidebar title */
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: #ad4e76 !important;
    font-family: Georgia, serif;
}

/* Sidebar labels */
section[data-testid="stSidebar"] label {
    color: #745965 !important;
    font-weight: 600;
}

/* Select boxes */
div[data-baseweb="select"] > div {
    background: #fffafd;
    border: 1px solid #e5b4c8;
    border-radius: 13px;
}

/* Text input */
.stTextInput input,
.stTextArea textarea {
    background-color: #fffafd !important;
    border: 1px solid #e4b5c8 !important;
    border-radius: 14px !important;
    color: #5f4c55 !important;
}

/* Input focus */
.stTextInput input:focus,
.stTextArea textarea:focus {
    border-color: #c96d92 !important;
    box-shadow: 0 0 0 2px rgba(201, 109, 146, 0.12) !important;
}

/* Placeholder */
textarea::placeholder,
input::placeholder {
    color: #b28d9b !important;
}

/* Generate button */
.stButton > button {
    width: 100%;
    border-radius: 15px;
    border: 1px solid #c9698e;
    background: linear-gradient(
        135deg,
        #d8799d,
        #b95d83
    );
    color: white;
    font-weight: 700;
    font-size: 16px;
    padding: 0.7rem 1rem;
    transition: all 0.3s ease;
}

/* Button hover */
.stButton > button:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 20px rgba(190, 90, 130, 0.25);
}

/* Download button */
.stDownloadButton > button {
    width: 100%;
    border-radius: 14px;
    border: 1px solid #d8a5ba;
    background: #fff4f8;
    color: #a94f75;
    font-weight: 700;
    transition: 0.3s;
}

.stDownloadButton > button:hover {
    background: #fce5ee;
    border-color: #c96d92;
}

/* Generated image */
img {
    border-radius: 20px;
    box-shadow:
        0 8px 25px rgba(160, 90, 120, 0.15);
}

/* Alerts */
div[data-testid="stAlert"] {
    border-radius: 15px;
}

/* Spinner */
.stSpinner > div {
    border-top-color: #c9698e !important;
}

/* Sidebar divider */
hr {
    border-color: #edcbd8 !important;
}

/* Footer */
.tulip-footer {
    text-align: center;
    margin-top: 30px;
    color: #a9788c;
    font-family: Georgia, serif;
    font-size: 14px;
}

</style>

<div class="tulip-decoration">
    🌷 🌷 🌷 🌷 🌷
</div>
""", unsafe_allow_html=True)


MODELS = {
    "FLUX.1 Schnell (fast, high quality)": "black-forest-labs/FLUX.1-schnell",
    "Stable Diffusion XL": "stabilityai/stable-diffusion-xl-base-1.0",
}


STYLES = {
    "None": "",
    "Photorealistic": ", photorealistic, 8k, highly detailed, sharp focus",
    "Anime": ", anime style, vibrant colors, studio ghibli inspired",
    "Digital Art": ", digital art, trending on artstation, concept art",
    "Oil Painting": ", oil painting, textured brush strokes, classical art",
    "3D Render": ", 3d render, octane render, soft lighting",
}


# ---------------- Sidebar ----------------
st.sidebar.header("🌷 Settings")

token = st.sidebar.text_input(
    "Hugging Face Token",
    value=os.getenv("HF_TOKEN", ""),
    type="password",
    help="Get a free token at https://huggingface.co/settings/tokens",
)

model_name = st.sidebar.selectbox(
    "Model",
    list(MODELS.keys())
)

style = st.sidebar.selectbox(
    "Style",
    list(STYLES.keys())
)

size = st.sidebar.selectbox(
    "Size",
    [
        "1024x1024",
        "768x768",
        "512x512",
        "1024x768",
        "768x1024"
    ]
)

width, height = map(int, size.split("x"))


# ---------------- Main ----------------
st.title("🎨 Text to Image Generator")

st.write(
    "Describe an image and let AI draw it. 🌷"
)


prompt = st.text_area(
    "Your prompt",
    placeholder="A cozy tea stall on a rainy evening in Hyderabad, warm lights, cinematic",
    height=100,
)

negative = st.text_input(
    "Negative prompt (optional)",
    placeholder="blurry, low quality, distorted"
)


if st.button(
    "✨ Generate",
    type="primary"
):

    if not token:

        st.error(
            "Please add your Hugging Face token in the sidebar "
            "(or in a .env file)."
        )

    elif not prompt.strip():

        st.warning(
            "Please enter a prompt."
        )

    else:

        client = InferenceClient(
            token=token
        )

        full_prompt = (
            prompt.strip()
            + STYLES[style]
        )

        kwargs = {
            "width": width,
            "height": height
        }

        if negative.strip():

            kwargs["negative_prompt"] = (
                negative.strip()
            )

        with st.spinner(
            "🌷 Generating your image... (10-60 seconds)"
        ):

            try:

                image = client.text_to_image(
                    full_prompt,
                    model=MODELS[model_name],
                    **kwargs
                )

            except Exception as e:

                st.error(
                    f"Generation failed: {e}"
                )

                st.info(
                    "Common fixes: check your token, "
                    "wait a minute if the model is loading, "
                    "or try the other model / a smaller size."
                )

                st.stop()


        st.image(
            image,
            caption=prompt
        )

        buf = BytesIO()

        image.save(
            buf,
            format="PNG"
        )

        st.download_button(
            "⬇️ Download PNG",
            buf.getvalue(),
            "generated.png",
            "image/png"
        )


# ---------------- Footer ----------------
st.markdown(
    """
    <div class="tulip-footer">
        🌷 Created with AI • Hugging Face • Streamlit 🌷
    </div>
    """,
    unsafe_allow_html=True
)

