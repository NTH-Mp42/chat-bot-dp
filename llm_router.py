"""
LLM Router dùng chung cho server.py (production) và app.py (Streamlit dev tool),
để không phải duy trì 2 bản logic gọi API khác nhau.

Thử lần lượt provider theo config.PROVIDER_ORDER (mặc định: gemini -> groq).
"""
import logging
from dataclasses import dataclass

from google import genai
from google.genai import types as genai_types
from openai import OpenAI  # Groq dùng endpoint tương thích OpenAI

import config

logger = logging.getLogger("llm_router")

RAG_SYSTEM_INSTRUCTION = """
Bạn là chatbot hỗ trợ học tập và cung cấp thông tin về trường THPT Hoài Đức A.

Khi trả lời:
- Chỉ sử dụng thông tin có trong NGỮ CẢNH được cung cấp.
- Không tự bịa hoặc suy đoán thông tin.
- Được phép thực hiện phép tính và suy luận trực tiếp từ thông tin trong NGỮ CẢNH trừ năm cuối cùng được đề cập.
- Khi cần tính toán theo thời gian, hãy sử dụng NĂM HIỆN TẠI được cung cấp trong prompt.
- Không lấy năm cuối cùng xuất hiện trong NGỮ CẢNH làm năm hiện tại.
- Ví dụ: nếu trường thành lập năm 1966 thì năm 2026 là 60 năm.
- Nếu câu hỏi hỏi về một mốc cụ thể, hãy tính theo đúng mốc đó.
- Nếu NGỮ CẢNH không đủ thông tin để trả lời, hãy nói rõ rằng hệ thống chưa có thông tin này.
- Trả lời ngắn gọn, rõ ràng, phù hợp với học sinh.
"""
DIRECT_SYSTEM_INSTRUCTION = """
Bạn là Chatbot hỗ trợ học tập và cung cấp thông tin trường học.

- Trả lời tự nhiên, thân thiện và ngắn gọn.
- Có thể trả lời các câu chào hỏi, cảm ơn và giao tiếp thông thường.
- Với câu hỏi kiến thức đơn giản, hãy cố gắng giải thích rõ ràng.
- Không tự nhận rằng mình có thông tin chính thức của nhà trường nếu thông tin đó chưa được cung cấp.
- Nếu câu hỏi yêu cầu thông tin cụ thể của nhà trường nhưng không có dữ liệu, hãy nói rằng hệ thống chưa có thông tin.
"""


@dataclass
class LLMResult:
    text: str
    provider: str
    model: str


class ProviderError(Exception):
    pass


def _call_gemini(prompt: str, system_instruction: str) -> LLMResult:
    if not config.GEMINI_API_KEY:
        raise ProviderError("Chưa cấu hình GEMINI_API_KEY")

    try:
        client = genai.Client(api_key=config.GEMINI_API_KEY)

        response = client.models.generate_content(
            model=config.GEMINI_MODEL,
            contents=prompt,
            config=genai_types.GenerateContentConfig(
                system_instruction=system_instruction
            ),
        )

        if not response.text:
            raise ProviderError("Gemini trả về nội dung rỗng")

        return LLMResult(
            text=response.text,
            provider="google",
            model=config.GEMINI_MODEL
        )

    except ProviderError:
        raise

    except Exception as e:
        raise ProviderError(
            f"Gemini lỗi: {type(e).__name__}: {e}"
        ) from e
def _call_groq(prompt: str, system_instruction: str) -> LLMResult:
    if not config.GROQ_API_KEY:
        raise ProviderError("Chưa cấu hình GROQ_API_KEY")

    try:
        client = OpenAI(
            api_key=config.GROQ_API_KEY,
            base_url="https://api.groq.com/openai/v1"
        )

        response = client.chat.completions.create(
            model=config.GROQ_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": system_instruction
                },
                {
                    "role": "user",
                    "content": prompt
                },
            ],
            temperature=0.2,
        )

        text = response.choices[0].message.content

        if not text:
            raise ProviderError("Groq trả về nội dung rỗng")

        return LLMResult(
            text=text,
            provider="groq",
            model=config.GROQ_MODEL
        )

    except ProviderError:
        raise

    except Exception as e:
        raise ProviderError(
            f"Groq lỗi: {type(e).__name__}: {e}"
        ) from e


_PROVIDER_FUNCS = {
    "gemini": _call_gemini,
    "groq": _call_groq
}

def generate(prompt: str, mode: str = "rag") -> LLMResult:
    """
    Gọi các provider theo config.PROVIDER_ORDER.

    mode:
        - "rag": dùng RAG_SYSTEM_INSTRUCTION
        - "direct": dùng DIRECT_SYSTEM_INSTRUCTION

    Raise RuntimeError nếu tất cả provider đều lỗi.
    """

    if mode == "direct":
        system_instruction = DIRECT_SYSTEM_INSTRUCTION
    elif mode == "rag":
        system_instruction = RAG_SYSTEM_INSTRUCTION
    else:
        raise ValueError(f"Mode không hợp lệ: {mode}")

    errors = []

    for name in config.PROVIDER_ORDER:
        func = _PROVIDER_FUNCS.get(name)

        if func is None:
            logger.warning(f"Provider không xác định: {name}")
            continue

        try:
            result = func(prompt, system_instruction)

            if name != config.PROVIDER_ORDER[0]:
                logger.info(f"Đã fallback sang provider: {name}")

            return result

        except ProviderError as e:
            logger.warning(f"Provider '{name}' thất bại: {e}")
            errors.append(str(e))

    raise RuntimeError(
        "Tất cả provider đều lỗi:\n" + "\n".join(errors)
    )