from typing import Optional
from pydantic import BaseModel, Field


class AudioAnalyzeParams(BaseModel):
    device_id: Optional[str] = Field(None, description="Identifier of the UAV/aircraft device")
    environment: Optional[str] = Field(None, description="Recording environment, e.g. 'indoor', 'outdoor'")
    operating_condition: Optional[str] = Field(None, description="Operating condition, e.g. 'hover', 'cruise'")
    rpm: Optional[float] = Field(None, description="Rotor/engine RPM at time of recording, if known")
    model_version: Optional[str] = Field(None, description="Force a specific model version instead of the active one")


class AudioMetadata(BaseModel):
    filename: str
    duration_seconds: float
    sample_rate: int
    channels: int
    format: str
    size_bytes: int
