import asyncio
import json
from datetime import datetime, timezone
import os
import base64
import re
import tempfile
import random

from fastapi import FastAPI, HTTPException, Request, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any, Union, Tuple
import time
import uuid
import logging

from gemini_webapi import GeminiClient, set_log_level
from gemini_webapi.constants import Model

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
set_log_level("INFO")

app = FastAPI(title="Gemini API FastAPI Server")

# Add CORS middleware
app.add_middleware(
	CORSMiddleware,
	allow_origins=["*"],
	allow_credentials=True,
	allow_methods=["*"],
	allow_headers=["*"],
)

# Global client and credential management
gemini_client = None
credential_index = -1  # 设置为-1，确保第一次调用get_next_credentials时返回索引0
failed_credentials = set()

# Authentication credentials
# Format for multiple credentials: "cred1|cred2|cred3"
SECURE_1PSID_LIST = os.environ.get("SECURE_1PSID", "").split("|")
SECURE_1PSIDTS_LIST = os.environ.get("SECURE_1PSIDTS", "").split("|")
API_KEY = os.environ.get("API_KEY", "")
PASSWORD = os.environ.get("PASSWORD", "")
EXPECTED_PASSWORD = "keliang"  # 程序中设定的固定密码

# Print debug info at startup
if not SECURE_1PSID_LIST or not SECURE_1PSID_LIST[0]:
	logger.warning("⚠️ Gemini API credentials are not set or empty! Please check your environment variables.")
	logger.warning("Make sure SECURE_1PSID and SECURE_1PSIDTS are correctly set in your .env file or environment.")
	logger.warning("For multiple credentials, use pipe (|) as separator: cred1|cred2|cred3")
	logger.warning("Example format in .env file (no quotes):")
	logger.warning("SECURE_1PSID=your_secure_1psid_value_here|another_1psid_value")
	logger.warning("SECURE_1PSIDTS=your_secure_1psidts_value_here|another_1psidts_value")
else:
	logger.info(f"Found {len(SECURE_1PSID_LIST)} credential pairs.")
	for i in range(len(SECURE_1PSID_LIST)):
		# 只显示PSID的前几个字符，但显示完整的PSIDTS（它更短且更容易区分）
		logger.info(f"Credential pair {i}: SECURE_1PSID starts with {SECURE_1PSID_LIST[i][:5]}..., SECURE_1PSIDTS: {SECURE_1PSIDTS_LIST[i]}")
	
	# Ensure the two credential lists have the same length
	if len(SECURE_1PSID_LIST) != len(SECURE_1PSIDTS_LIST):
		logger.warning("⚠️ The number of SECURE_1PSID and SECURE_1PSIDTS credentials don't match!")
		# Trim to the shorter list
		min_length = min(len(SECURE_1PSID_LIST), len(SECURE_1PSIDTS_LIST))
		SECURE_1PSID_LIST = SECURE_1PSID_LIST[:min_length]
		SECURE_1PSIDTS_LIST = SECURE_1PSIDTS_LIST[:min_length]
		logger.warning(f"Using only the first {min_length} credential pairs.")

if not API_KEY:
	logger.warning("⚠️ API_KEY is not set or empty! API authentication will not work.")
	logger.warning("Make sure API_KEY is correctly set in your .env file or environment.")
else:
	logger.info(f"API_KEY found. API_KEY starts with: {API_KEY[:5]}...")

if not PASSWORD:
	logger.warning("⚠️ PASSWORD is not set! API will not work until PASSWORD is set to the correct value.")
	logger.warning(f"Expected PASSWORD: {EXPECTED_PASSWORD}")
elif PASSWORD != EXPECTED_PASSWORD:
	logger.warning(f"⚠️ PASSWORD is set but does not match the expected password ({EXPECTED_PASSWORD})! API will not work.")
	logger.warning("Cookie has expired. Please refresh your browser and try again.")
else:
	logger.info("Password protection is enabled and password is correct.")

# Credential rotation lock to prevent race conditions
credential_lock = asyncio.Lock()

async def get_next_credentials() -> Tuple[str, str]:
	"""Get the next valid credential pair using round-robin approach with failure tracking"""
	global credential_index, failed_credentials
	
	async with credential_lock:
		if not SECURE_1PSID_LIST or not SECURE_1PSID_LIST[0]:
			return "", ""
			
		# If all credentials have failed, reset and try again
		if len(failed_credentials) >= len(SECURE_1PSID_LIST):
			logger.warning("All credentials have failed. Resetting failed tracking and trying again.")
			failed_credentials = set()
		
		# Find a credential that hasn't failed
		for _ in range(len(SECURE_1PSID_LIST)):
			credential_index = (credential_index + 1) % len(SECURE_1PSID_LIST)
			if credential_index not in failed_credentials:
				logger.info(f"Selected credential pair {credential_index} for use")
				return SECURE_1PSID_LIST[credential_index], SECURE_1PSIDTS_LIST[credential_index]
		
		# If we get here, all credentials have failed
		return "", ""

async def mark_credential_failed(index: int):
	"""Mark a credential as failed"""
	global failed_credentials
	
	async with credential_lock:
		failed_credentials.add(index)
		logger.warning(f"Marked credential pair {index} as failed. Total failed: {len(failed_credentials)}/{len(SECURE_1PSID_LIST)}")


def correct_markdown(md_text: str) -> str:
	"""
	修正Markdown文本，移除Google搜索链接包装器，并根据显示文本简化目标URL。
	"""
	def simplify_link_target(text_content: str) -> str:
		match_colon_num = re.match(r"([^:]+:\d+)", text_content)
		if match_colon_num:
			return match_colon_num.group(1)
		return text_content

	def replacer(match: re.Match) -> str:
		outer_open_paren = match.group(1)
		display_text = match.group(2)

		new_target_url = simplify_link_target(display_text)
		new_link_segment = f"[`{display_text}`]({new_target_url})"

		if outer_open_paren:
			return f"{outer_open_paren}{new_link_segment})"
		else:
			return new_link_segment
	pattern = r"(\()?\[`([^`]+?)`\]\((https://www.google.com/search\?q=)(.*?)(?<!\\)\)\)*(\))?"
	
	fixed_google_links = re.sub(pattern, replacer, md_text)
	# fix wrapped markdownlink
	pattern = r"`(\[[^\]]+\]\([^\)]+\))`"
	return re.sub(pattern, r'\1', fixed_google_links)


# Pydantic models for API requests and responses
class ContentItem(BaseModel):
	type: str
	text: Optional[str] = None
	image_url: Optional[Dict[str, str]] = None


class Message(BaseModel):
	role: str
	content: Union[str, List[ContentItem]]
	name: Optional[str] = None


class ChatCompletionRequest(BaseModel):
	model: str
	messages: List[Message]
	temperature: Optional[float] = 0.7
	top_p: Optional[float] = 1.0
	n: Optional[int] = 1
	stream: Optional[bool] = False
	max_tokens: Optional[int] = None
	presence_penalty: Optional[float] = 0
	frequency_penalty: Optional[float] = 0
	user: Optional[str] = None


class Choice(BaseModel):
	index: int
	message: Message
	finish_reason: str


class Usage(BaseModel):
	prompt_tokens: int
	completion_tokens: int
	total_tokens: int


class ChatCompletionResponse(BaseModel):
	id: str
	object: str = "chat.completion"
	created: int
	model: str
	choices: List[Choice]
	usage: Usage


class ModelData(BaseModel):
	id: str
	object: str = "model"
	created: int
	owned_by: str = "google"


class ModelList(BaseModel):
	object: str = "list"
	data: List[ModelData]


# Authentication dependency
async def verify_api_key(authorization: str = Header(None)):
	# Check if the environment password is correct
	if not PASSWORD or PASSWORD != EXPECTED_PASSWORD:
		logger.warning("环境变量中的密码未设置或不正确")
		raise HTTPException(status_code=401, detail="Cookie has expired. Please refresh your browser and try again.")
	
	if not API_KEY:
		# If API_KEY is not set in environment, skip validation (for development)
		logger.warning("API key validation skipped - no API_KEY set in environment")
		return

	if not authorization:
		raise HTTPException(status_code=401, detail="Missing Authorization header")
	
	try:
		scheme, token = authorization.split()
		if scheme.lower() != "bearer":
			raise HTTPException(status_code=401, detail="Invalid authentication scheme. Use Bearer token")
		
		if token != API_KEY:
			raise HTTPException(status_code=401, detail="Invalid API key")
	except ValueError:
		raise HTTPException(status_code=401, detail="Invalid authorization format. Use 'Bearer YOUR_API_KEY'")
	
	return token


# Simple error handler middleware
@app.middleware("http")
async def error_handling(request: Request, call_next):
	try:
		return await call_next(request)
	except Exception as e:
		logger.error(f"Request failed: {str(e)}")
		return JSONResponse(status_code=500, content={"error": {"message": str(e), "type": "internal_server_error"}})


# Get list of available models
@app.get("/v1/models")
async def list_models():
	"""返回 gemini_webapi 中声明的模型列表"""
	# Check if the environment password is correct
	if not PASSWORD or PASSWORD != EXPECTED_PASSWORD:
		logger.warning("环境变量中的密码未设置或不正确")
		raise HTTPException(status_code=401, detail="Cookie has expired. Please refresh your browser and try again.")
		
	now = int(datetime.now(tz=timezone.utc).timestamp())
	data = [
		{
			"id": m.model_name,  # 如 "gemini-2.0-flash"
			"object": "model",
			"created": now,
			"owned_by": "google-gemini-web",
		}
		for m in Model
	]
	print(data)
	return {"object": "list", "data": data}


# Helper to convert between Gemini and OpenAI model names
def map_model_name(openai_model_name: str) -> Model:
	"""根据模型名称字符串查找匹配的 Model 枚举值"""
	# 打印所有可用模型以便调试
	all_models = [m.model_name if hasattr(m, "model_name") else str(m) for m in Model]
	logger.info(f"Available models: {all_models}")

	# 首先尝试直接查找匹配的模型名称
	for m in Model:
		model_name = m.model_name if hasattr(m, "model_name") else str(m)
		if openai_model_name.lower() in model_name.lower():
			return m

	# 如果找不到匹配项，使用默认映射
	model_keywords = {
		"gemini-pro": ["pro", "2.0"],
		"gemini-pro-vision": ["vision", "pro"],
		"gemini-flash": ["flash", "2.0"],
		"gemini-1.5-pro": ["1.5", "pro"],
		"gemini-1.5-flash": ["1.5", "flash"],
	}

	# 根据关键词匹配
	keywords = model_keywords.get(openai_model_name, ["pro"])  # 默认使用pro模型

	for m in Model:
		model_name = m.model_name if hasattr(m, "model_name") else str(m)
		if all(kw.lower() in model_name.lower() for kw in keywords):
			return m

	# 如果还是找不到，返回第一个模型
	return next(iter(Model))


# Prepare conversation history from OpenAI messages format
def prepare_conversation(messages: List[Message]) -> tuple:
	conversation = ""
	temp_files = []

	for msg in messages:
		if isinstance(msg.content, str):
			# String content handling
			if msg.role == "system":
				conversation += f"System: {msg.content}\n\n"
			elif msg.role == "user":
				conversation += f"Human: {msg.content}\n\n"
			elif msg.role == "assistant":
				conversation += f"Assistant: {msg.content}\n\n"
		else:
			# Mixed content handling
			if msg.role == "user":
				conversation += "Human: "
			elif msg.role == "system":
				conversation += "System: "
			elif msg.role == "assistant":
				conversation += "Assistant: "

			for item in msg.content:
				if item.type == "text":
					conversation += item.text or ""
				elif item.type == "image_url" and item.image_url:
					# Handle image
					image_url = item.image_url.get("url", "")
					if image_url.startswith("data:image/"):
						# Process base64 encoded image
						try:
							# Extract the base64 part
							base64_data = image_url.split(",")[1]
							image_data = base64.b64decode(base64_data)

							# Create temporary file to hold the image
							with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
								tmp.write(image_data)
								temp_files.append(tmp.name)
						except Exception as e:
							logger.error(f"Error processing base64 image: {str(e)}")

			conversation += "\n\n"

	# Add a final prompt for the assistant to respond to
	conversation += "Assistant: "

	return conversation, temp_files


# Dependency to get the initialized Gemini client
async def get_gemini_client():
	global gemini_client, credential_index
	
	if gemini_client is None:
		try:
			# Get the first set of credentials
			secure_1psid, secure_1psidts = await get_next_credentials()
			if not secure_1psid or not secure_1psidts:
				logger.error("No valid credentials available")
				raise HTTPException(status_code=500, detail="No valid credentials available")
				
			logger.info(f"Initializing Gemini client with credential pair {credential_index}")
			gemini_client = GeminiClient(secure_1psid, secure_1psidts)
			await gemini_client.init(timeout=300)
			logger.info(f"Gemini client initialized successfully with credential pair {credential_index}")
		except Exception as e:
			logger.error(f"Failed to initialize Gemini client with credential pair {credential_index}: {str(e)}")
			# Mark the current credential as failed
			await mark_credential_failed(credential_index)
			# Try with the next credential
			return await try_with_new_credentials()
	
	return gemini_client

async def try_with_new_credentials():
	"""Try to initialize a new client with different credentials"""
	global gemini_client, credential_index
	
	try:
		# Get new credentials
		secure_1psid, secure_1psidts = await get_next_credentials()
		
		if not secure_1psid or not secure_1psidts:
			logger.error("No valid credentials available")
			raise HTTPException(status_code=500, detail="No valid credentials available")
		
		# Create a new client
		logger.info(f"Trying to initialize new Gemini client with credential pair {credential_index}")
		gemini_client = GeminiClient(secure_1psid, secure_1psidts)
		await gemini_client.init(timeout=300)
		logger.info(f"Initialized new Gemini client with credential pair {credential_index}")
		return gemini_client
	except Exception as e:
		logger.error(f"Failed to initialize new Gemini client with credential pair {credential_index}: {str(e)}")
		# Mark the current credential as failed
		await mark_credential_failed(credential_index)
		# Try with the next credential if available
		if len(failed_credentials) < len(SECURE_1PSID_LIST):
			return await try_with_new_credentials()
		else:
			raise HTTPException(status_code=500, detail="All credentials failed")

@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest, api_key: str = Depends(verify_api_key)):
	try:
		# 确保客户端已初始化
		global gemini_client, credential_index
		if gemini_client is None:
			secure_1psid, secure_1psidts = await get_next_credentials()
			if not secure_1psid or not secure_1psidts:
				raise HTTPException(status_code=500, detail="No valid credentials available")
				
			gemini_client = GeminiClient(secure_1psid, secure_1psidts)
			await gemini_client.init(timeout=300)
			logger.info(f"Gemini client initialized successfully with credential pair {credential_index}")

		# 转换消息为对话格式
		conversation, temp_files = prepare_conversation(request.messages)
		logger.info(f"Prepared conversation: {conversation}")
		logger.info(f"Temp files: {temp_files}")

		# 获取适当的模型
		model = map_model_name(request.model)
		logger.info(f"Using model: {model}")

		# 尝试使用当前凭证生成响应
		max_retries = min(3, len(SECURE_1PSID_LIST))
		retry_count = 0
		
		while retry_count < max_retries:
			try:
				logger.info(f"Sending request to Gemini using credential pair {credential_index}...")
				
				if temp_files:
					# With files
					response = await gemini_client.generate_content(conversation, files=temp_files, model=model)
				else:
					# Text only
					response = await gemini_client.generate_content(conversation, model=model)
				
				# 如果成功，继续处理
				break
			except Exception as e:
				retry_count += 1
				logger.warning(f"Request failed with credential pair {credential_index}: {str(e)}")
				
				if retry_count < max_retries:
					# 尝试使用新凭证
					gemini_client = await try_with_new_credentials()
					if not gemini_client:
						raise HTTPException(status_code=500, detail="Failed to initialize with new credentials")
				else:
					# 所有重试都失败
					raise HTTPException(status_code=500, detail=f"All credential attempts failed: {str(e)}")

		# 清理临时文件
		for temp_file in temp_files:
			try:
				os.unlink(temp_file)
			except Exception as e:
				logger.warning(f"Failed to delete temp file {temp_file}: {str(e)}")

		# 提取文本响应
		reply_text = ""
		if hasattr(response, "text"):
			reply_text = response.text
		else:
			reply_text = str(response)
		reply_text = reply_text.replace("&lt;","<").replace("\\<","<").replace("\\_","_").replace("\\>",">")
		reply_text = correct_markdown(reply_text)
		
		logger.info(f"Response: {reply_text}")

		if not reply_text or reply_text.strip() == "":
			logger.warning("Empty response received from Gemini")
			reply_text = "服务器返回了空响应。请检查 Gemini API 凭据是否有效。"

		# 创建响应对象
		completion_id = f"chatcmpl-{uuid.uuid4()}"
		created_time = int(time.time())

		# 每次请求后自动轮换到下一个凭证
		# 这样可以确保凭证被均匀使用，避免一个凭证被过度使用
		try:
			# 在请求成功处理后，预先切换到下一个凭证
			logger.info("Rotating to next credential for future requests")
			new_secure_1psid, new_secure_1psidts = await get_next_credentials()
			if new_secure_1psid and new_secure_1psidts:
				gemini_client = GeminiClient(new_secure_1psid, new_secure_1psidts)
				await gemini_client.init(timeout=300)
				logger.info(f"Rotated to credential pair {credential_index} for next request")
		except Exception as e:
			# 如果轮换失败，记录错误但不影响当前请求的响应
			logger.warning(f"Failed to rotate credentials: {str(e)}")

		# 检查客户端是否请求流式响应
		if request.stream:
			# 实现流式响应
			async def generate_stream():
				# 创建 SSE 格式的流式响应
				# 先发送开始事件
				data = {
					"id": completion_id,
					"object": "chat.completion.chunk",
					"created": created_time,
					"model": request.model,
					"choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
				}
				yield f"data: {json.dumps(data)}\n\n"

				# 模拟流式输出 - 将文本按字符分割发送
				for char in reply_text:
					data = {
						"id": completion_id,
						"object": "chat.completion.chunk",
						"created": created_time,
						"model": request.model,
						"choices": [{"index": 0, "delta": {"content": char}, "finish_reason": None}],
					}
					yield f"data: {json.dumps(data)}\n\n"
					# 可选：添加短暂延迟以模拟真实的流式输出
					await asyncio.sleep(0.01)

				# 发送结束事件
				data = {
					"id": completion_id,
					"object": "chat.completion.chunk",
					"created": created_time,
					"model": request.model,
					"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
				}
				yield f"data: {json.dumps(data)}\n\n"
				yield "data: [DONE]\n\n"

			return StreamingResponse(generate_stream(), media_type="text/event-stream")
		else:
			# 非流式响应（原来的逻辑）
			result = {
				"id": completion_id,
				"object": "chat.completion",
				"created": created_time,
				"model": request.model,
				"choices": [{"index": 0, "message": {"role": "assistant", "content": reply_text}, "finish_reason": "stop"}],
				"usage": {
					"prompt_tokens": len(conversation.split()),
					"completion_tokens": len(reply_text.split()),
					"total_tokens": len(conversation.split()) + len(reply_text.split()),
				},
			}

			logger.info(f"Returning response: {result}")
			return result

	except Exception as e:
		logger.error(f"Error generating completion: {str(e)}", exc_info=True)
		raise HTTPException(status_code=500, detail=f"Error generating completion: {str(e)}")


@app.get("/")
async def root():
	# Check if the environment password is correct
	if not PASSWORD or PASSWORD != EXPECTED_PASSWORD:
		logger.warning("环境变量中的密码未设置或不正确")
		raise HTTPException(status_code=401, detail="Cookie has expired. Please refresh your browser and try again.")
		
	return {"status": "online", "message": "Gemini API FastAPI Server is running"}


if __name__ == "__main__":
	import uvicorn

	uvicorn.run("main:app", host="0.0.0.0", port=8000, log_level="info")
