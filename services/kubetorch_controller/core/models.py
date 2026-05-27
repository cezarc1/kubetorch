"""
Pydantic models for Kubetorch Controller API.
"""

from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel


# ============================================================================
# Pool Specifier Models
# ============================================================================


class LabelSelectorSpecifier(BaseModel):
    """Specifier for tracking pods via K8s label selector."""

    type: Literal["label_selector"] = "label_selector"
    selector: Dict[str, str]  # e.g. {"app": "workers", "team": "ml"}


# ============================================================================
# Pool Service Config Models
# ============================================================================


class ServiceConfigUrl(BaseModel):
    """User-provided service URL (e.g. for Knative)."""

    url: str


class ServiceConfigSelector(BaseModel):
    """Custom selector for routing (e.g. Ray head node only)."""

    selector: Dict[str, str]


class ServiceConfigName(BaseModel):
    """Custom service name."""

    name: str


ServiceConfig = Union[ServiceConfigUrl, ServiceConfigSelector, ServiceConfigName, None]


# ============================================================================
# Runtime & Distributed Config Models
# ============================================================================


class RuntimeConfig(BaseModel):
    """Runtime configuration that flows to pods via WebSocket."""

    log_streaming_enabled: Optional[bool] = True
    metrics_enabled: Optional[bool] = True
    inactivity_ttl: Optional[str] = None
    log_level: Optional[str] = None
    allowed_serialization: Optional[str] = None


class PoolMetadata(BaseModel):
    """Structured metadata stored with each pool.

    Contains user info and configuration that flows to pods.
    Stored in the database as JSON in the pool_metadata column.
    """

    username: Optional[str] = None
    deployment_mode: Optional[str] = None
    distributed_config: Optional[Dict[str, Any]] = None
    runtime_config: Optional[RuntimeConfig] = None


# ============================================================================
# Pool Module Models
# ============================================================================


class ModulePointers(BaseModel):
    """Pointer information to locate and load a callable on the pod."""

    file_path: Optional[str] = None
    module_name: Optional[str] = None
    cls_or_fn_name: Optional[str] = None
    project_root: Optional[str] = None
    init_args: Optional[Dict[str, Any]] = None


class PoolModule(BaseModel):
    """A module deployed onto a pool."""

    type: Literal["fn", "cls", "cmd", "app"]  # Function, class, command, or app
    pointers: ModulePointers  # Spec to load or run the module
    dispatch: Literal["regular", "spmd", "load_balanced"] = "regular"
    procs: Union[int, List[int]] = 1  # Number of processes or specific indices


# ============================================================================
# Pool Request/Response Models
# ============================================================================


class PoolRequest(BaseModel):
    """Request for the /pool endpoint.

    Registers a compute pool: a logical group of pods that calls can be directed to.
    """

    # Required fields
    name: str  # Unique identifier for the pool
    namespace: str
    specifier: LabelSelectorSpecifier  # Label selector to track pods

    # Optional fields
    service: Optional[ServiceConfig] = None  # How to route calls to the pool
    dockerfile: Optional[str] = None  # Instructions to rebuild workers
    module: Optional[PoolModule] = None  # Application deployed on the pool
    pool_metadata: Optional[PoolMetadata] = None  # username, runtime_config, etc.

    # Service configuration for auto-created services
    server_port: Optional[int] = 32300
    labels: Optional[Dict[str, Any]] = None
    annotations: Optional[Dict[str, Any]] = None

    # K8s resource info - used for teardown to know what to delete
    resource_kind: Optional[
        str
    ] = None  # e.g., "Deployment", "StatefulSet", "PyTorchJob"
    resource_name: Optional[
        str
    ] = None  # Name of the K8s resource (defaults to pool name)

    # Whether to create a headless service for distributed pod discovery (SPMD)
    create_headless_service: bool = False


class PoolResponse(BaseModel):
    """Response from the /pool endpoint.

    Contains all pool data - consistent across GET, POST, and list operations.
    """

    name: str
    namespace: str
    status: str
    message: str
    service_url: Optional[str] = None  # URL to reach the pool
    pod_ips: Optional[List[str]] = None  # List of pod IPs
    specifier: Optional[Dict[str, Any]] = None  # Pool specifier (selector)

    # Additional fields for consistency with list_pools and PoolRequest
    service_config: Optional[Dict[str, Any]] = None
    dockerfile: Optional[str] = None
    module: Optional[Dict[str, Any]] = None
    pool_metadata: Optional[Dict[str, Any]] = None
    server_port: Optional[int] = None
    labels: Optional[Dict[str, Any]] = None
    annotations: Optional[Dict[str, Any]] = None
    resource_kind: Optional[str] = None  # e.g., "Deployment", "RayCluster"
    resource_name: Optional[str] = None  # Name of the K8s resource
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    last_deployed_at: Optional[str] = None


class ReadinessResponse(BaseModel):
    """Response from check-ready endpoint."""

    ready: bool
    message: str
    resource_type: str
    details: Optional[Dict[str, Any]] = None


# ============================================================================
# Apply Request/Response Models
# ============================================================================


class ApplyRequest(BaseModel):
    """Request for the /apply endpoint.

    Applies a K8s compute manifest (like kubectl apply).
    Creates pods/resources in the cluster.
    """

    service_name: str
    namespace: str
    resource_type: str
    resource_manifest: dict


class ApplyResponse(BaseModel):
    """Response from the /apply endpoint."""

    service_name: str
    namespace: str
    resource_type: str
    status: str
    message: str
    resource: Optional[Any] = None  # The created K8s resource (if available)


# ============================================================================
# Deploy Request/Response Models
# ============================================================================


class DeployRequest(BaseModel):
    """Request for the /deploy endpoint.

    Combines apply (create K8s resource) and pool registration.
    """

    # From ApplyRequest
    service_name: str
    namespace: str
    resource_type: str
    resource_manifest: dict

    # From PoolRequest
    specifier: LabelSelectorSpecifier
    service: Optional[ServiceConfig] = None
    dockerfile: Optional[str] = None
    module: Optional[PoolModule] = None
    pool_metadata: Optional[PoolMetadata] = None
    server_port: Optional[int] = 32300
    labels: Optional[Dict[str, Any]] = None
    annotations: Optional[Dict[str, Any]] = None
    create_headless_service: bool = False


class DeployResponse(BaseModel):
    """Response from the /deploy endpoint.

    Contains results from both apply and pool registration operations.
    """

    # Apply result
    service_name: str
    namespace: str
    resource_type: str
    apply_status: str
    apply_message: str
    resource: Optional[Any] = None

    # Pool result
    pool_status: str
    pool_message: str
    service_url: Optional[str] = None
    resource_kind: Optional[str] = None
    resource_name: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    last_deployed_at: Optional[str] = None


# ============================================================================
# Run Request/Response Models
# ============================================================================


class RunCreateRequest(BaseModel):
    run_id: str
    namespace: str
    command: List[str]
    source_key: str
    author: Optional[str] = None
    intent: Optional[str] = None
    logs_key: Optional[str] = None
    image: Optional[str] = None
    resources: Optional[Dict[str, Any]] = None
    env: Optional[Dict[str, Any]] = None
    job_name: Optional[str] = None
    labels: Optional[Dict[str, Any]] = None
    annotations: Optional[Dict[str, Any]] = None


class RunStatusUpdateRequest(BaseModel):
    status: str
    exit_code: Optional[int] = None


class RunLogsUpdateRequest(BaseModel):
    logs: str


class RunNoteCreateRequest(BaseModel):
    body: str
    author: Optional[str] = None


class RunArtifactCreateRequest(BaseModel):
    name: str
    uri: str
    kind: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    author: Optional[str] = None


class RunNoteResponse(BaseModel):
    id: int
    run_id: str
    author: Optional[str] = None
    body: str
    created_at: Optional[str] = None


class RunArtifactResponse(BaseModel):
    id: int
    run_id: str
    name: str
    uri: str
    kind: Optional[str] = None
    metadata: Dict[str, Any] = {}
    author: Optional[str] = None
    created_at: Optional[str] = None


class RunResponse(BaseModel):
    id: int
    run_id: str
    namespace: str
    author: Optional[str] = None
    intent: Optional[str] = None
    command: List[str]
    status: str
    exit_code: Optional[int] = None
    source_key: str
    logs_key: Optional[str] = None
    image: Optional[str] = None
    resources: Dict[str, Any] = {}
    env: Dict[str, Any] = {}
    job_name: Optional[str] = None
    labels: Dict[str, Any] = {}
    annotations: Dict[str, Any] = {}
    created_at: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    updated_at: Optional[str] = None
    notes: List[RunNoteResponse] = []
    artifacts: List[RunArtifactResponse] = []


class RunListResponse(BaseModel):
    runs: List[RunResponse]


class RunDeleteResponse(BaseModel):
    run_id: str
    deleted_run: bool
    deleted_notes: int
    deleted_artifacts: int
    deleted_job: bool
    job_name: Optional[str] = None


class RunLogsUpdateResponse(BaseModel):
    run_id: str
    logs_bytes: int


# ============================================================================
# Delete Resource Request/Response Models
# ============================================================================
class ServiceTeardownRequest(BaseModel):
    namespace: str
    services: Optional[Union[dict, str, list]] = None
    force: Optional[bool] = False
    prefix: Optional[str] = None
    teardown_all: Optional[bool] = False
    username: Optional[str] = None
    exact_match: Optional[bool] = False
