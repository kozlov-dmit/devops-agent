import requests
import base64

def gigachat_token() -> str:
    url = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"

    payload={
    'scope': 'GIGACHAT_API_PERS'
    }
    headers = {
    'Content-Type': 'application/x-www-form-urlencoded',
    'Accept': 'application/json',
    'RqUID': 'd1d43c6c-e091-470d-8730-4be2a493f653',
    'Authorization': 'Basic MGFlN2NiMjYtM2RmYy00NDkzLWI0OWUtNjgzOTdkNTdhYzUzOjQxMTUyZmFiLTYxN2ItNGM2Ni1hMzUyLWM2OWQ4ZGU3MmJkYw=='
    }

    response = requests.request("POST", url, headers=headers, data=payload, verify=False)

    return response.json()["access_token"]

def base64_credentials(plain_credentials: str) -> bytes:
    return base64.b64encode(plain_credentials.encode('utf-8')).decode()

print(base64_credentials("0ae7cb26-3dfc-4493-b49e-68397d57ac53:b8452c7d-85b6-4398-b035-1f35e4fa5831"))
