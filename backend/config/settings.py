from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    debug: bool = False
    groq_api_key: str
    firebase_credentials_path:str
    model_config = SettingsConfigDict(env_file=".env")

settings = Settings()