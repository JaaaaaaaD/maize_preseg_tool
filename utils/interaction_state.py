class InteractionStateMachine:
    """Small state machine for top-level interaction modes."""

    IDLE = "idle"
    PREANNOTATION_BOX = "preannotation_box"
    PREANNOTATION_CANDIDATE = "preannotation_candidate"
    FINE_TUNE = "fine_tune"
    FINE_TUNE_ADD_VERTEX = "fine_tune_add_vertex"
    FINE_TUNE_DELETE_VERTEX = "fine_tune_delete_vertex"
    FINE_TUNE_SPLIT_STAGING = "fine_tune_split_staging"
    IGNORE_REGION = "ignore_region"
    REMOVAL_REGION = "removal_region"

    _ALLOWED_TRANSITIONS = {
        IDLE: {
            PREANNOTATION_BOX,
            PREANNOTATION_CANDIDATE,
            FINE_TUNE,
            IGNORE_REGION,
            REMOVAL_REGION,
        },
        PREANNOTATION_BOX: {IDLE, PREANNOTATION_CANDIDATE},
        PREANNOTATION_CANDIDATE: {IDLE, FINE_TUNE, PREANNOTATION_BOX},
        FINE_TUNE: {IDLE, FINE_TUNE_ADD_VERTEX, FINE_TUNE_DELETE_VERTEX, FINE_TUNE_SPLIT_STAGING, REMOVAL_REGION},
        FINE_TUNE_ADD_VERTEX: {FINE_TUNE, FINE_TUNE_DELETE_VERTEX, FINE_TUNE_SPLIT_STAGING, IDLE},
        FINE_TUNE_DELETE_VERTEX: {FINE_TUNE, FINE_TUNE_ADD_VERTEX, FINE_TUNE_SPLIT_STAGING, IDLE},
        FINE_TUNE_SPLIT_STAGING: {FINE_TUNE, FINE_TUNE_ADD_VERTEX, FINE_TUNE_DELETE_VERTEX, IDLE},
        IGNORE_REGION: {IDLE},
        REMOVAL_REGION: {IDLE, FINE_TUNE},
    }

    def __init__(self):
        self.state = self.IDLE

    def can_transition(self, next_state):
        if next_state == self.state:
            return True
        return next_state in self._ALLOWED_TRANSITIONS.get(self.state, set())

    def transition(self, next_state):
        if not self.can_transition(next_state):
            return False
        self.state = next_state
        return True

    def force(self, next_state):
        self.state = next_state
        return self.state
