class OptimizationCoordinator:
    def __init__(self):
        self.all_evaluations = []  # In-memory
        self.checkpoint_interval = 50  # Every 50 evals
        self.checkpoint_time_interval = 300  # Every 5 minutes

    def collect_result(self, result: EvaluationResult):
        self.all_evaluations.append(result)

        # Checkpoint by count
        if len(self.all_evaluations) % self.checkpoint_interval == 0:
            self.save_checkpoint()

        # Checkpoint by time (check elapsed)
        if time.time() - self.last_checkpoint_time > self.checkpoint_time_interval:
            self.save_checkpoint()