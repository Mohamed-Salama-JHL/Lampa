from bokeh.embed import server_document
from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from starlette.responses import HTMLResponse
import user_agent
import os

app = FastAPI()
templates = Jinja2Templates(directory="./APIs/templates")

@app.get("/test_hello")
async def root():
    return {"message": "Hello World"}


@app.get("/")
async def bkapp_page(request: Request):
    dashboard_url = os.environ['dashboard_url']
    script = server_document(dashboard_url)
    return templates.TemplateResponse("new_base.html", {"request": request, "script": script})





