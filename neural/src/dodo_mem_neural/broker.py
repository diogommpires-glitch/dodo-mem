"""Authenticated local evidence broker; callers cannot choose their own scope."""

import asyncio
from inspect import signature
from time import monotonic
from typing import Annotated, Literal
from uuid import uuid4

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from .contracts import Credential, EvidencePacket, ModelRoute, authenticate
from .resources import ResourcePressure
from .retrieval import SourceChanged, _render


class RetrievalRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str = Field(min_length=1, max_length=2000)
    model: str = Field(min_length=1, max_length=200)
    node_id: str | None = Field(default=None, min_length=1, max_length=256)


class ValidationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    packet_id: str = Field(min_length=1, max_length=64)
    model: str = Field(min_length=1, max_length=200)
    source_count: int | None = Field(default=None, ge=0, le=8)


class ContextRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str = Field(min_length=1, max_length=2000)
    node_id: str | None = Field(default=None, min_length=1, max_length=256)


class BackgroundRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    paused: bool


class ActivityControl(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    visible: bool
    paused: bool
    token_index: int | None = Field(default=None, ge=0, le=255)


def create_broker_app(
    engine,
    credentials: tuple[Credential, ...],
    *,
    routes: tuple[ModelRoute, ...],
    status=None,
    topics=None,
    background_control=None,
    activity=None,
    projection_source=None,
):
    app = FastAPI(title="Dodo memory broker", docs_url=None, redoc_url=None)
    scoped_status = status is not None and "scope" in signature(status).parameters
    models = {route.id: route for route in routes}
    packets = {}
    topic_jobs = {}
    weight_job = None
    if len(models) != len(routes):
        raise ValueError("model route IDs must be unique")

    def principal(request: Request):
        try:
            return authenticate(request.headers.get("authorization"), credentials)
        except PermissionError as error:
            raise HTTPException(401, str(error)) from error

    def activity_owner(credential: Annotated[Credential, Depends(principal)]):
        if credential.client != "dodo" or credential.scope != "full_private":
            raise HTTPException(403, "Activation inspection is local-owner only.")
        if activity is None:
            raise HTTPException(503, "Neural activity is unavailable.")
        return credential

    def snapshot():
        return (
            projection_source.activity_snapshot()
            if hasattr(projection_source, "activity_snapshot")
            else activity.snapshot()
        )

    @app.get("/activity/architecture", dependencies=[Depends(activity_owner)])
    async def activity_architecture():
        try:
            result = (
                await asyncio.to_thread(projection_source.activity_architecture)
                if hasattr(projection_source, "activity_architecture")
                else activity.architecture()
            )
        except (OSError, RuntimeError, ValueError) as error:
            raise HTTPException(503, "Verified E5 architecture metadata is unavailable.") from error
        if result["model"].get("name") != "e5" or result.get("verified") is not True:
            raise HTTPException(503, "The active E5 architecture is not verified.")
        return result

    @app.get("/activity/snapshot", dependencies=[Depends(activity_owner)])
    async def activity_snapshot():
        return snapshot()

    @app.post("/activity/control", dependencies=[Depends(activity_owner)])
    async def activity_control(body: ActivityControl):
        activity.control(**body.model_dump())
        return snapshot()

    @app.get("/activity/projections", dependencies=[Depends(activity_owner)])
    async def projection_descriptors():
        if projection_source is None:
            raise HTTPException(503, "Static E5 parameters are unavailable.")
        try:
            return await asyncio.to_thread(projection_source.projection_descriptors)
        except (OSError, RuntimeError, ValueError) as error:
            raise HTTPException(503, "Verified E5 parameter metadata is unavailable.") from error

    @app.get("/activity/weights", dependencies=[Depends(activity_owner)])
    async def projection_weights(
        block: int = Query(default=11, ge=0, le=11),
        projection: Literal[
            "query", "key", "value", "attention_output", "ff_up", "ff_down"
        ] = "ff_down",
        limit: int = Query(default=512, ge=1, le=512),
        output_index: int | None = Query(default=None, ge=0, le=1535),
    ):
        nonlocal weight_job
        if projection_source is None:
            raise HTTPException(503, "Static E5 parameters are unavailable.")
        output_size = 1536 if projection == "ff_up" else 384
        if output_index is not None and output_index >= output_size:
            raise HTTPException(422, "Output feature is outside this projection.")
        if weight_job is not None and not weight_job.done():
            raise HTTPException(429, "Another parameter view is loading. Retry when it finishes.")
        weight_job = asyncio.create_task(
            asyncio.to_thread(
                projection_source.projection_weights, block, projection, limit, output_index
            )
        )
        weight_job.add_done_callback(lambda done: None if done.cancelled() else done.exception())
        try:
            # A cancelled HTTP request must not admit a second live binary reader.
            return await asyncio.shield(weight_job)
        except ResourcePressure as error:
            raise HTTPException(503, str(error)) from error
        except (OSError, RuntimeError, ValueError) as error:
            raise HTTPException(503, "Verified E5 parameter data is unavailable.") from error

    @app.get("/activity/activations/{capture_id}", dependencies=[Depends(activity_owner)])
    async def activation_values(capture_id: str):
        if len(capture_id) > 64 or (result := activity.capture(capture_id)) is None:
            raise HTTPException(404, "Activation capture expired or is unavailable.")
        return result

    @app.get("/activity/events", dependencies=[Depends(activity_owner)])
    async def activity_events(request: Request, after: int = Query(default=0, ge=0)):
        async def stream():
            import json

            cursor = after
            yield "event: snapshot\ndata: " + json.dumps(snapshot()) + "\n\n"
            while not await request.is_disconnected():
                batch = activity.events_since(cursor)
                if batch["gap"]:
                    cursor = batch["snapshot"]["sequence"]
                    yield "event: gap\ndata: " + json.dumps(snapshot()) + "\n\n"
                else:
                    for event in batch["events"]:
                        cursor = event["sequence"]
                        yield (
                            "id: "
                            + str(cursor)
                            + "\nevent: "
                            + event["type"]
                            + "\ndata: "
                            + json.dumps(event)
                            + "\n\n"
                        )
                await asyncio.sleep(0.1)

        return StreamingResponse(
            stream(), media_type="text/event-stream", headers={"Cache-Control": "no-store"}
        )

    @app.post("/index/control")
    async def index_control(
        body: BackgroundRequest, credential: Annotated[Credential, Depends(principal)]
    ):
        if credential.client != "dodo" or credential.scope != "full_private":
            raise HTTPException(403, "Background control is local-owner only.")
        if background_control is None:
            raise HTTPException(503, "Background control is unavailable.")
        return background_control(body.paused)

    def remember(packet, credential, model):
        stale = [key for key, value in packets.items() if monotonic() - value[3] > 120]
        for key in stale:
            del packets[key]
        while len(packets) >= 64:
            del packets[next(iter(packets))]
        packet_id = uuid4().hex
        packets[packet_id] = (credential.token_sha256, model, packet, monotonic())
        return {**packet.as_dict(), "packet_id": packet_id}

    @app.get("/status")
    async def broker_status(credential: Annotated[Credential, Depends(principal)]):
        result = (
            status(scope=credential.scope)
            if scoped_status
            else status()
            if status
            else {"status": "ready"}
        )
        if credential.scope == "full_private" and hasattr(engine, "diagnostics"):
            diagnostic = engine.diagnostics
            result = {
                **result,
                "retrieval_diagnostic": {
                    "stage": diagnostic["stage"],
                    "seconds": monotonic() - diagnostic["started"],
                    "error_type": diagnostic["error_type"],
                },
            }
        return {**result, "client": credential.client, "scope": credential.scope}

    @app.get("/paths")
    async def paths(credential: Annotated[Credential, Depends(principal)]):
        try:
            return engine.paths(credential.scope)
        except SourceChanged as error:
            raise HTTPException(
                409, "source_changed: retry retrieval against current evidence"
            ) from error

    @app.get("/topics")
    async def broker_topics(credential: Annotated[Credential, Depends(principal)]):
        if topics is None:
            return {"topics": [], "counts": {}}
        job = topic_jobs.get(credential.scope)
        if job is None or job.done():
            job = asyncio.create_task(asyncio.to_thread(topics, credential.scope))
            topic_jobs[credential.scope] = job
            job.add_done_callback(lambda done: None if done.cancelled() else done.exception())
        # A caller timeout must not leave another untracked SQLite reader behind.
        return await asyncio.shield(job)

    @app.post("/retrieve")
    async def retrieve(
        body: RetrievalRequest, credential: Annotated[Credential, Depends(principal)]
    ):
        if body.model not in models or body.model not in credential.routes:
            raise HTTPException(403, "model route is not authorised for this credential")
        try:
            options = {"node_id": body.node_id} if body.node_id is not None else {}
            packet = await engine.retrieve(body.query, credential.scope, body.model, **options)
        except PermissionError as error:
            raise HTTPException(
                403, "selected node is unavailable or outside the credential scope"
            ) from error
        except SourceChanged as error:
            raise HTTPException(
                409, "source_changed: retry retrieval against current evidence"
            ) from error
        except (OSError, RuntimeError, ValueError) as error:
            if hasattr(engine, "diagnostics"):
                engine.diagnostics["error_type"] = type(error).__name__
            raise HTTPException(
                503, "memory_unavailable: retry or explicitly disable memory"
            ) from error
        return remember(packet, credential, body.model)

    @app.post("/context")
    async def context(body: ContextRequest, credential: Annotated[Credential, Depends(principal)]):
        try:
            options = {"node_id": body.node_id} if body.node_id is not None else {}
            packet = await engine.retrieve(
                body.query, credential.scope, "local-inspection", **options
            )
        except PermissionError as error:
            raise HTTPException(
                403, "selected node is unavailable or outside the credential scope"
            ) from error
        except SourceChanged as error:
            raise HTTPException(
                409, "source_changed: retry retrieval against current evidence"
            ) from error
        except (OSError, RuntimeError, ValueError) as error:
            if hasattr(engine, "diagnostics"):
                engine.diagnostics["error_type"] = type(error).__name__
            raise HTTPException(
                503, "memory_unavailable: retry local context inspection"
            ) from error
        return {
            **remember(packet, credential, "local-inspection"),
            "coverage": {
                **packet.coverage,
                "token_count_mode": "utf8-byte-upper-bound",
                "dispatch_allowed": False,
            },
        }

    @app.post("/validate")
    async def validate(
        body: ValidationRequest, credential: Annotated[Credential, Depends(principal)]
    ):
        receipt = packets.get(body.packet_id)
        if (
            not receipt
            or receipt[:2] != (credential.token_sha256, body.model)
            or monotonic() - receipt[3] > 120
        ):
            raise HTTPException(
                409, "memory packet expired or does not belong to this credential and route"
            )
        try:
            engine.verify(receipt[2], credential.scope)
        except SourceChanged as error:
            raise HTTPException(
                409, "source_changed: retry retrieval against current evidence"
            ) from error
        if body.source_count is not None:
            packet = receipt[2]
            sources = packet.sources[: body.source_count]
            selected = EvidencePacket(
                _render(sources), sources, {**packet.coverage, "returned": len(sources)}
            )
            packets[body.packet_id] = (*receipt[:2], selected, receipt[3])
        return {"valid": True}

    @app.post("/packet")
    async def packet(
        body: ValidationRequest, credential: Annotated[Credential, Depends(principal)]
    ):
        await validate(body, credential)
        return {**packets[body.packet_id][2].as_dict(), "packet_id": body.packet_id}

    return app
