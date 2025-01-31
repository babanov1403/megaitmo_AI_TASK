import time
from typing import List

from fastapi import FastAPI, HTTPException, Request, Response
from pydantic import HttpUrl
from schemas.request import PredictionRequest, PredictionResponse
from utils.logger import setup_logger

import torch
import os

import requests_utils
import ml_tools

tavily_token = os.getenv('tavily_token')
api_key = os.getenv('api_key')
folder_id = os.getenv('folder_id')


# Initialize
app = FastAPI()


model = ml_tools.InitializeModel(folder_id, api_key)

logger = None


@app.on_event("startup")
async def startup_event():
    global logger
    logger = await setup_logger()


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    body = await request.body()
    await logger.info(
        f"Incoming request: {request.method} {request.url}\n"
        f"Request body: {body.decode()}"
    )

    response = await call_next(request)
    process_time = time.time() - start_time

    response_body = b""
    async for chunk in response.body_iterator:
        response_body += chunk

    await logger.info(
        f"Request completed: {request.method} {request.url}\n"
        f"Status: {response.status_code}\n"
        f"Response body: {response_body.decode()}\n"
        f"Duration: {process_time:.3f}s"
    )

    return Response(
        content=response_body,
        status_code=response.status_code,
        headers=dict(response.headers),
        media_type=response.media_type,
    )


@app.post("/api/request", response_model=PredictionResponse)
async def predict(body: PredictionRequest):
    try:
        await logger.info(f"Processing prediction request with id: {body.id}")

        question = body.query
        flag = ml_tools.IsEnumerated(question)
        rephrased_question = ml_tools.FromQuestionMakeQuery(question=question, model = model)
        print(rephrased_question)
        url_list = requests_utils.GetURLs(rephrased_question, tavily_token)

        big_text = ""

        for url in url_list:
            scraped = requests_utils.GetInfoFromURL(url, rephrased_question, model)
            if scraped == None:
                continue
            big_text += scraped + ".\n"
            if len(big_text) >= 10000:
                break
        # summary = ml_tools.Summarize(big_text, model)
        answer = ml_tools.GiveAnswerWithContext(big_text, question, model)
        is_valid = ml_tools.ValidateAnwser(answer, model)
        answer_number = ml_tools.DefineAnswerNumber(answer, question, model)
        if not flag:
            answer_number = '-1'
        else:
            try:
                if int(answer_number) == -1:
                    answer_number = '-1'
                elif int(answer_number) > 10:
                    answer_number = '-1'

            except ValueError:
                answer_number = '3'

        sources: List[HttpUrl] = []

        for url in url_list[:3]:
            sources.append(HttpUrl(url))

        # Здесь будет вызов вашей модели
        # answer = 1  # Замените на реальный вызов модели
        # sources: List[HttpUrl] = [
        #     HttpUrl("https://itmo.ru/ru/"),
        #     HttpUrl("https://abit.itmo.ru/"),
        # ]
        answer += " Ответ был дан YaGPT.PRO."
        response = PredictionResponse(
            id=body.id,
            answer=answer_number,
            reasoning=answer,
            sources=sources,
        )
        await logger.info(f"Successfully processed request {body.id}")
        return response
    except ValueError as e:
        error_msg = str(e)
        await logger.error(f"Validation error for request {body.id}: {error_msg}")
        raise HTTPException(status_code=400, detail=error_msg)
    except Exception as e:
        await logger.error(f"Internal error processing request {body.id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")
