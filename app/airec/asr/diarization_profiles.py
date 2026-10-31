"""Реестр гиперпараметров диаризации: группы настроек и внешний YAML-конфиг.

Диаризация работает по схеме «движок + группа параметров»:

- у каждого движка (PyAnnote Segmentation, WeSpeaker ResNet34) есть свой
  набор калиброванных групп — «Совещание», «Интервью», «Конференция»;
  группу выбирает пользователь, движок — реестр по текстовому ID модели;
- группы хранятся во внешнем файле ``diarization_profiles.yaml`` в папке
  данных приложения.  Файл перечитывается при каждом запуске стадии, поэтому
  правки применяются без перезапуска программы;
- файл отсутствует или повреждён -> используется встроенный набор дефолтов
  (:data:`DEFAULT_PROFILES`), файл пересоздаётся при первом запуске стадии,
  чтобы его можно было открыть и отредактировать;
- калиброванные настройки Silero VAD и связки с GigaAM (компенсация задержки
  CTC и допуск границ слов) хранятся в секциях ``vad`` и ``gigaam`` того же
  файла и тоже применяются на лету.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

log = logging.getLogger(__name__)

#: Имя внешнего конфигурационного файла (и его JSON-альтернативы) в папке данных.
PROFILES_FILE_NAME = "diarization_profiles.yaml"
PROFILES_FILE_NAME_JSON = "diarization_profiles.json"

#                      ГРУППЫ PYANNOTE SEGMENTATION 3.0
# Powerset-бинаризация: кадр считается активным при вероятности >= onset и
# остаётся активным, пока вероятность не упадёт до offset (гистерезис
# исключает мерцание метки на паузах).
DEFAULT_PYANNOTE_PROFILES: list[dict[str, Any]] = [
    {
        "id": "meeting",
        "label": "Совещание (несколько голосов)",
        "onset": 0.54,
        "offset": 0.42,
        "min_duration_on": 0.12,
        "min_duration_off": 0.22,
        "pad_onset": 0.05,
        "pad_offset": 0.05,
        "window_duration": 10.0,
        "window_step": 3.5,
    },
    {
        "id": "interview",
        "label": "Интервью (чистый диалог)",
        "onset": 0.50,
        "offset": 0.38,
        "min_duration_on": 0.20,
        "min_duration_off": 0.35,
        "pad_onset": 0.05,
        "pad_offset": 0.05,
        "window_duration": 10.0,
        "window_step": 5.0,
    },
    {
        "id": "conference",
        "label": "Конференция (динамический зал)",
        "onset": 0.60,
        "offset": 0.46,
        "min_duration_on": 0.22,
        "min_duration_off": 0.20,
        "pad_onset": 0.05,
        "pad_offset": 0.05,
        "window_duration": 10.0,
        "window_step": 5.0,
    },
]

#                     ГРУППЫ WESPEAKER RESNET34 (эмбеддинги + кластеризация)
DEFAULT_WESPEAKER_PROFILES: list[dict[str, Any]] = [
    {
        "id": "meeting",
        "label": "Совещание (несколько голосов)",
        "method": "centroid",
        "threshold": 0.74,
        "min_cluster_size": 16,
    },
    {
        "id": "interview",
        "label": "Интервью (чистый диалог)",
        "method": "average",
        "threshold": 0.69,
        "min_cluster_size": 22,
    },
    {
        "id": "conference",
        "label": "Конференция (динамический зал)",
        "method": "centroid",
        "threshold": 0.79,
        "min_cluster_size": 16,
    },
]

#                     КАЛИБРОВАННЫЙ БЛОК SILERO VAD ONNX
# Silero формирует первичные границы аудио, которые поступают и в GigaAM, и в
# диаризацию; от этих значений зависит, не отсекаются ли русские «да/ага/так»
# и не дробится ли предложение на куски.
DEFAULT_VAD: dict[str, Any] = {
    # Порог вероятности речи (в шумном зале стоит поднять до 0.58).
    "threshold": 0.50,
    # Выше 200 мс отсекаются короткие «да/ага/так».
    "min_speech_duration_ms": 100,
    # Ниже 150 мс русское предложение дробится на несколько сегментов.
    "min_silence_duration_ms": 220,
    # Защита взрывных [п, т, к]: 60 мс тишины перед началом звука.
    "speech_pad_ms": 60,
    # Справочное поле: модель Silero v5 на 16 кГц использует фреймы 512 отсчётов
    # (32 мс); переопределить его через onnx-asr нельзя.
    "window_size_samples": 512,
}

#                  СВЯЗКА С GIGAAM V3 E2E CTC INT8
DEFAULT_GIGAAM: dict[str, Any] = {
    # Физическая задержка CTC-пика Conformer относительно фонемы: точка
    # голосования слова в диаризации сдвигается вперёд на это значение.
    "ctc_delay_compensation_ms": 70,
    # Временной допуск при сопоставлении слова с окнами диаризации: токен,
    # попавший в зазор между регионами, привязывается к ближайшему краю,
    # если он не дальше этого допуска.
    "word_boundary_margin_ms": 80,
    # Зарезервировано: порог IoU для присвоения слова двум спикерам при
    # перекрытии.  В текущей версии не применяется (формат результата
    # подразумевает один спикер на сегмент), ключ сохранён для совместимости.
    "overlap_attribution_threshold": 0.45,
}

DEFAULT_PROFILES: dict[str, Any] = {
    "pyannote": {"profiles": [dict(p) for p in DEFAULT_PYANNOTE_PROFILES]},
    "wespeaker": {"profiles": [dict(p) for p in DEFAULT_WESPEAKER_PROFILES]},
    "vad": dict(DEFAULT_VAD),
    "gigaam": dict(DEFAULT_GIGAAM),
}


class PyannoteGroup(BaseModel):
    """Группа гиперпараметров PyAnnote Segmentation (PowerSet-бинаризация)."""

    model_config = ConfigDict(extra="ignore")

    id: str = Field(min_length=1, max_length=48)
    label: str = Field(min_length=1, max_length=96)
    # Активация кадра (гистерезис: включение по onset, выключение по offset).
    onset: float = Field(default=0.5, ge=0.05, le=0.95)
    offset: float = Field(default=0.5, ge=0.01, le=0.95)
    # Речь короче этого отбрасывается; пауза короче этого склеивается.
    min_duration_on: float = Field(default=0.20, ge=0.02, le=5.0)
    min_duration_off: float = Field(default=0.0, ge=0.0, le=5.0)
    # Защита согласных: регион расширяется на pad_onset влево и pad_offset вправо.
    pad_onset: float = Field(default=0.0, ge=0.0, le=2.0)
    pad_offset: float = Field(default=0.0, ge=0.0, le=2.0)
    # Скользящее окно сегментации (секунды).
    window_duration: float = Field(default=10.0, ge=5.0, le=30.0)
    window_step: float = Field(default=5.0, ge=1.0, le=30.0)

    @model_validator(mode="after")
    def _check_thresholds(self) -> PyannoteGroup:
        if self.offset > self.onset:
            raise ValueError("offset должен быть не больше onset (гистерезис)")
        if self.window_step > self.window_duration:
            raise ValueError("window_step не может превышать window_duration")
        return self


class WeSpeakerGroup(BaseModel):
    """Группа гиперпараметров WeSpeaker (эмбеддинги + кластеризация)."""

    model_config = ConfigDict(extra="ignore")

    id: str = Field(min_length=1, max_length=48)
    label: str = Field(min_length=1, max_length=96)
    # "centroid" — сравнение с центроидом кластера (ко-цепочка), "average" —
    # сравнение со средним попарным сходством с членами кластера.
    method: Literal["centroid", "average"] = "centroid"
    threshold: float = Field(default=0.70, ge=0.1, le=0.99)
    # Кластеры, набравшие меньше сегментов, приклеиваются к ближайшему.
    min_cluster_size: int = Field(default=16, ge=1, le=10_000)


class VadGroup(BaseModel):
    """Калиброванные настройки Silero VAD ONNX."""

    model_config = ConfigDict(extra="ignore")

    threshold: float = Field(default=0.50, ge=0.10, le=0.99)
    min_speech_duration_ms: int = Field(default=100, ge=30, le=2000)
    min_silence_duration_ms: int = Field(default=220, ge=40, le=5000)
    speech_pad_ms: int = Field(default=60, ge=0, le=500)
    # Справочное поле (Silero v5 использует 512 отсчётов при 16 кГц).
    window_size_samples: int = Field(default=512)


class GigaamLink(BaseModel):
    """Настройки связки GigaAM v3 CTC с окнами диаризации."""

    model_config = ConfigDict(extra="ignore")

    ctc_delay_compensation_ms: int = Field(default=70, ge=0, le=500)
    word_boundary_margin_ms: int = Field(default=80, ge=0, le=1000)
    overlap_attribution_threshold: float = Field(default=0.45, ge=0.0, le=1.0)


class DiarizationProfiles(BaseModel):
    """Полный набор групп для всех движков + калибровки VAD и GigaAM."""

    model_config = ConfigDict(extra="ignore")

    pyannote: list[PyannoteGroup]
    wespeaker: list[WeSpeakerGroup]
    vad: VadGroup = Field(default_factory=VadGroup)
    gigaam: GigaamLink = Field(default_factory=GigaamLink)

    def pyannote_group(self, group_id: str | None) -> PyannoteGroup:
        """Группа PyAnnote по ID (``None`` — первая, она же активная по умолчанию)."""
        if not group_id:
            return self.pyannote[0]
        for group in self.pyannote:
            if group.id == group_id:
                return group
        raise KeyError(group_id)

    def wespeaker_group(self, group_id: str | None) -> WeSpeakerGroup:
        """Группа WeSpeaker по ID (``None`` — первая)."""
        if not group_id:
            return self.wespeaker[0]
        for group in self.wespeaker:
            if group.id == group_id:
                return group
        raise KeyError(group_id)


def _merge_group_list(
    defaults: list[dict[str, Any]],
    user_raw: list[Any] | None,
) -> list[dict[str, Any]]:
    """Merge user group definitions over the built-in set.

    Группы пользователя с тем же ID заменяют поля дефолтной группы; новые ID
    добавляются как пользовательские группы (например, свой профиль под
    конкретный микрофон).
    """
    if not isinstance(user_raw, list):
        return defaults
    merged = {g.get("id"): {**g} for g in defaults if isinstance(g, dict) and g.get("id")}
    for raw in user_raw:
        if not isinstance(raw, dict):
            continue
        group_id = raw.get("id")
        if not isinstance(group_id, str) or not group_id:
            continue
        base = merged.get(group_id, {"id": group_id})
        base.update(raw)
        merged[group_id] = base
    return list(merged.values())


def _profiles_from_payload(payload: Any) -> DiarizationProfiles:
    """Build a validated set of profiles from an arbitrary config payload."""
    data = {**DEFAULT_PROFILES}
    if isinstance(payload, dict):
        for section in ("pyannote", "wespeaker"):
            raw = payload.get(section)
            if isinstance(raw, dict) and isinstance(raw.get("profiles"), list):
                data[section] = {"profiles": raw["profiles"]}
            elif isinstance(raw, list):
                data[section] = {"profiles": raw}
        if isinstance(payload.get("vad"), dict):
            data["vad"] = {**DEFAULT_VAD, **payload["vad"]}
        if isinstance(payload.get("gigaam"), dict):
            data["gigaam"] = {**DEFAULT_GIGAAM, **payload["gigaam"]}

    return DiarizationProfiles(
        pyannote=[
            PyannoteGroup.model_validate(item)
            for item in _merge_group_list(
                DEFAULT_PYANNOTE_PROFILES,
                data["pyannote"]["profiles"],
            )
        ],
        wespeaker=[
            WeSpeakerGroup.model_validate(item)
            for item in _merge_group_list(
                DEFAULT_WESPEAKER_PROFILES,
                data["wespeaker"]["profiles"],
            )
        ],
        vad=VadGroup.model_validate(data["vad"]),
        gigaam=GigaamLink.model_validate(data["gigaam"]),
    )


def _read_payload(path: Path) -> Any:
    """Parse YAML or JSON; anything unparsable raises so the caller falls back."""
    text = path.read_text(encoding="utf-8-sig")
    if path.suffix.lower() == ".json":
        return json.loads(text)
    return yaml.safe_load(text)


def _write_default_file(path: Path) -> None:
    """Create an editable YAML file with the built-in defaults."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    payload = {
        "pyannote": {"profiles": DEFAULT_PYANNOTE_PROFILES},
        "wespeaker": {"profiles": DEFAULT_WESPEAKER_PROFILES},
        "vad": DEFAULT_VAD,
        "gigaam": DEFAULT_GIGAAM,
    }
    tmp.write_text(
        yaml.safe_dump(
            payload,
            allow_unicode=True,
            sort_keys=False,
            default_flow_style=False,
        ),
        encoding="utf-8",
    )
    os.replace(tmp, path)


# cache: parse only when the file actually changed on disk
_CACHE: dict[tuple[str, tuple[int, int, int]], DiarizationProfiles] = {}


def profiles_file_exists(data_dir: Path | str | None) -> bool:
    """Whether the user's diarization profiles file exists in *data_dir*.

    Нужен менеджеру, чтобы отличить «файл есть» от «работаем на встроенных
    дефолтах»: calibrated VAD-значения из файла перекрывают старые настройки
    VAD только когда файл действительно есть.
    """
    if data_dir is None:
        return False
    root = Path(data_dir)
    return (root / PROFILES_FILE_NAME).exists() or (root / PROFILES_FILE_NAME_JSON).exists()


def load_diarization_profiles(
    data_dir: Path | str | None = None, *, bootstrap: bool = True
) -> DiarizationProfiles:
    """Read the external profiles file (YAML/JSON) and return validated groups.

    Файл ищется в *data_dir* (папка данных приложения).  Отсутствует ->
    записывается файл с дефолтами (чтобы пользователь мог его отредактировать);
    повреждён -> предупреждение и дефолты.  Правки файла подхватываются
    автоматически: загрузчик кэширует результат по mtime и перечитывает файл
    при каждом вызове после изменения (без перезапуска приложения).
    """
    defaults = _profiles_from_payload(None)
    if data_dir is None:
        return defaults

    yaml_path = Path(data_dir) / PROFILES_FILE_NAME
    json_path = Path(data_dir) / PROFILES_FILE_NAME_JSON
    path = json_path if json_path.exists() else yaml_path

    try:
        stat = path.stat()
    except OSError:
        if bootstrap:
            try:
                _write_default_file(yaml_path)
            except OSError as exc:  # read-only data dir: keep working silently
                log.warning("cannot create %s: %s", yaml_path, exc)
        return defaults

    key = (str(path), (stat.st_mtime_ns, stat.st_size, stat.st_ino))
    cached = _CACHE.get(key)
    if cached is not None:
        return cached

    try:
        payload = _read_payload(path)
        profiles = _profiles_from_payload(payload)
    except (
        OSError,
        ValueError,
        TypeError,
        ValidationError,
        yaml.YAMLError,
    ) as exc:
        # Любой сбой файла (повреждён, чужая структура, недопустимые значения)
        # — это дефолты и предупреждение, а не падение стадии.
        log.warning("%s is broken or invalid (%s); using built-in defaults", path, exc)
        return defaults

    if len(_CACHE) > 8:
        _CACHE.clear()
    _CACHE[key] = profiles
    return profiles
