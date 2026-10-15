from typing import List, Tuple

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, model_validator

app = FastAPI(title="Customer Churn Prediction Brain", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChurnRequest(BaseModel):
    usedGB: float = Field(..., ge=0)
    totalGB: float = Field(..., gt=0)
    remainingGB: float = Field(..., ge=0)
    daysLeft: int = Field(..., ge=0)

    @model_validator(mode="after")
    def check_consistency(self):
        if self.usedGB > self.totalGB:
            raise ValueError("usedGB cannot exceed totalGB")
        return self


class ChurnResponse(BaseModel):
    churn_probability: int = Field(..., ge=0, le=100)
    risk_level: str
    reasons: List[str]
    recommendation: str


def score_churn(req: ChurnRequest) -> Tuple[int, List[str]]:
    score = 10
    reasons: List[str] = []

    usage_ratio = req.usedGB / req.totalGB if req.totalGB > 0 else 0
    remaining_ratio = req.remainingGB / req.totalGB if req.totalGB > 0 else 0

    # 1. Time pressure (exclusive tiers)
    if req.daysLeft <= 3:
        score += 40
        reasons.append(f"Only {req.daysLeft} days left (expiry imminent)")
    elif req.daysLeft <= 7:
        score += 25
        reasons.append(f"Only {req.daysLeft} days left")
    elif req.daysLeft <= 14:
        score += 10
        reasons.append(f"{req.daysLeft} days left until renewal")

    # 2. Stalled usage
    if usage_ratio < 0.2 and req.daysLeft < 15:
        score += 30
        reasons.append(
            f"Low consumption: only {usage_ratio:.0%} used with {req.daysLeft} days left"
        )
        if req.usedGB == 0:
            score += 15
            reasons.append("Zero usage detected - service never adopted")

    # 3. Heavy remainder near expiry
    if remaining_ratio > 0.7 and req.daysLeft <= 7:
        score += 20
        reasons.append(
            f"High remainder: {remaining_ratio:.0%} unused with {req.daysLeft} days left"
        )

    # 4. Healthy reducer
    if usage_ratio > 0.8 and req.daysLeft > 7:
        score -= 15
        reasons.append(f"Healthy usage: {usage_ratio:.0%} consumed, active user")

    if not reasons:
        reasons.append("Normal usage pattern")

    score = max(0, min(100, score))
    return score, reasons


def to_risk_level(score: int) -> Tuple[str, str]:
    if score > 65:
        return "high", "High churn risk - send renewal reminder / retention offer now."
    if score >= 30:
        return "medium", "Medium risk - monitor usage and nudge with usage tips."
    return "low", "Low risk - healthy customer, no action needed."


@app.get("/")
def root():
    return {"status": "ok", "service": "ml-brain"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict", response_model=ChurnResponse)
def predict(req: ChurnRequest):
    probability, reasons = score_churn(req)
    risk_level, recommendation = to_risk_level(probability)
    return ChurnResponse(
        churn_probability=probability,
        risk_level=risk_level,
        reasons=reasons,
        recommendation=recommendation,
    )
