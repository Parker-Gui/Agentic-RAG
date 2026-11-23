"""Authenticated Agent task and profile endpoints."""

from typing import Any
from uuid import UUID

from fastapi import APIRouter, Query, Response, status

from app.api.deps import AgentServiceDep, ArtifactServiceDep, CurrentUser
from app.schemas.agent import (
    AgentTaskCreate,
    AgentTaskList,
    AgentTaskRead,
    ConversationRead,
    ProfileMemoryCreate,
    ProfileMemoryList,
    ProfileMemoryUpdate,
    TaskStatus,
)
from app.schemas.artifact import ArtifactList, ArtifactRead

router = APIRouter()


@router.post("/tasks/{task_id}/artifacts", response_model=ArtifactRead, status_code=status.HTTP_201_CREATED)
async def create_artifact(task_id: UUID, user: CurrentUser, service: ArtifactServiceDep, response: Response) -> Any:
    metadata, created = await service.create(user.id, task_id)
    response.status_code = status.HTTP_201_CREATED if created else status.HTTP_200_OK
    return metadata


@router.get("/tasks/{task_id}/artifacts", response_model=ArtifactList)
async def list_artifacts(task_id: UUID, user: CurrentUser, service: ArtifactServiceDep) -> Any:
    return await service.list_for_task(user.id, task_id)


@router.get("/artifacts/{artifact_id}", response_model=ArtifactRead)
async def get_artifact(artifact_id: UUID, user: CurrentUser, service: ArtifactServiceDep) -> Any:
    return await service.get(user.id, artifact_id)


@router.get("/artifacts/{artifact_id}/download", response_class=Response)
async def download_artifact(artifact_id: UUID, user: CurrentUser, service: ArtifactServiceDep) -> Any:
    content, filename = await service.download(user.id, artifact_id)
    return Response(
        content=content,
        media_type="text/markdown; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.post("/tasks", response_model=AgentTaskRead, status_code=status.HTTP_201_CREATED)
async def run_task(payload: AgentTaskCreate, user: CurrentUser, service: AgentServiceDep) -> Any:
    return await service.run(user.id, payload.question, conversation_id=payload.conversation_id)


@router.get("/conversations/{conversation_id}", response_model=ConversationRead)
async def get_conversation(
    conversation_id: UUID,
    user: CurrentUser,
    service: AgentServiceDep,
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=100),
) -> Any:
    return await service.get_conversation(user.id, conversation_id, skip=skip, limit=limit)


@router.get("/tasks/{task_id}", response_model=AgentTaskRead)
async def get_task(task_id: UUID, user: CurrentUser, service: AgentServiceDep) -> Any:
    return await service.get_task(user.id, task_id)


@router.get("/tasks", response_model=AgentTaskList)
async def list_tasks(
    user: CurrentUser,
    service: AgentServiceDep,
    conversation_id: UUID | None = None,
    task_status: TaskStatus | None = Query(None, alias="status"),
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=100),
) -> Any:
    return await service.list_tasks(
        user.id, conversation_id=conversation_id, status=task_status, skip=skip, limit=limit
    )


@router.post("/tasks/{task_id}/pause", response_model=AgentTaskRead)
async def pause_task(task_id: UUID, user: CurrentUser, service: AgentServiceDep, response: Response) -> Any:
    result, requested = await service.pause(user.id, task_id)
    response.status_code = status.HTTP_202_ACCEPTED if requested else status.HTTP_200_OK
    return result


@router.post("/tasks/{task_id}/abandon", response_model=AgentTaskRead)
async def abandon_task(task_id: UUID, user: CurrentUser, service: AgentServiceDep) -> Any:
    return await service.abandon(user.id, task_id)


@router.post("/tasks/{task_id}/resume", response_model=AgentTaskRead)
async def resume_task(task_id: UUID, user: CurrentUser, service: AgentServiceDep) -> Any:
    return await service.resume(user.id, task_id)


@router.post("/profile", response_model=ProfileMemoryList, status_code=status.HTTP_201_CREATED)
async def remember(payload: ProfileMemoryCreate, user: CurrentUser, service: AgentServiceDep) -> Any:
    return {"items": await service.remember(user.id, payload.text)}


@router.patch("/profile/{memory_id}", response_model=dict[str, Any])
async def update_memory(
    memory_id: UUID, payload: ProfileMemoryUpdate, user: CurrentUser, service: AgentServiceDep
) -> Any:
    return await service.update_memory(user.id, memory_id, payload.text)


@router.delete("/profile/{memory_id}", status_code=status.HTTP_204_NO_CONTENT, response_model=None)
async def forget(memory_id: UUID, user: CurrentUser, service: AgentServiceDep) -> Any:
    await service.forget(user.id, memory_id)


@router.get("/profile/{memory_id}/history", response_model=list[dict[str, Any]])
async def memory_history(memory_id: UUID, user: CurrentUser, service: AgentServiceDep) -> Any:
    return await service.memory_history(user.id, memory_id)
