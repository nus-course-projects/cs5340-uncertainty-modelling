from dataclasses import dataclass
from typing import TypedDict, List


class MetadataDict(TypedDict):
  id: str
  org_text: str
  clean_text: str
  start_time: float
  end_time: float
  signer_id: int
  signer: int
  start: int
  end: int
  file: str
  label: int
  height: float
  width: float
  fps: float
  url: str
  text: str
  box: List[float]
  filename: str


@dataclass
class Metadata:
  id: str
  org_text: str
  clean_text: str
  start_time: float
  end_time: float
  signer_id: int
  signer: int
  start: int
  end: int
  file: str
  label: int
  height: float
  width: float
  fps: float
  url: str
  text: str
  box: List[float]
  filename: str

  def as_dict(self) -> MetadataDict:
    return {
      "id": self.id,
      "org_text": self.org_text,
      "clean_text": self.clean_text,
      "start_time": self.start_time,
      "end_time": self.end_time,
      "signer_id": self.signer_id,
      "signer": self.signer,
      "start": self.start,
      "end": self.end,
      "file": self.file,
      "label": self.label,
      "height": self.height,
      "width": self.width,
      "fps": self.fps,
      "url": self.url,
      "text": self.text,
      "box": self.box,
      "filename": self.filename
    }
