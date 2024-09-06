from nemo.utils import AppState


class SageMakerAppState(AppState):
    def __init__(self):
        super().__init__()

        # Replication process group
        self._current_replication_group = None
        self._replication_coordinator_rank = None

        # FSDP process group
        self._fsdp_process_group = None
        self._is_fsdp_action_rank = None
        self._fsdp_coordinator_rank = None

    @property
    def current_replication_group(self):
        return self._current_replication_group

    @current_replication_group.setter
    def current_replication_group(self, group):
        self._current_replication_group = group

    @property
    def replication_coordinator_rank(self):
        return self._replication_coordinator_rank

    @replication_coordinator_rank.setter
    def replication_coordinator_rank(self, rank):
        self._replication_coordinator_rank = rank

    @property
    def fsdp_process_group(self):
        return self._fsdp_process_group

    @fsdp_process_group.setter
    def fsdp_process_group(self, group):
        self._fsdp_process_group = group

    @property
    def is_fsdp_action_rank(self):
        return self._is_fsdp_action_rank

    @is_fsdp_action_rank.setter
    def is_fsdp_action_rank(self, is_action):
        self._is_fsdp_action_rank = is_action

    @property
    def fsdp_coordinator_rank(self):
        return self._fsdp_coordinator_rank

    @fsdp_coordinator_rank.setter
    def fsdp_coordinator_rank(self, rank):
        self._fsdp_coordinator_rank = rank
