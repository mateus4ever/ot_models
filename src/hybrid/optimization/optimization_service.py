class OptimizationService:
    """
    Unified optimization service
    Runs locally (threads) or remotely (k8s pods) - same interface
    """

    def __init__(self, config, execution_mode='local'):
        self.config = config
        self.execution_mode = execution_mode  # 'local' or 'cloud'
        self.jobs = {}  # job tracking

    def submit_optimization(self,
                            strategy: str,
                            param_space: Dict,
                            optimizer_type: OptimizerType,
                            n_workers: int = 16) -> str:
        """Submit optimization job (async, returns immediately)"""
        job_id = self._generate_job_id()

        optimizer = OptimizerFactory.create_optimizer(optimizer_type, config)
        optimizer.set_strategy_config(strategy, param_space)
        optimizer.set_n_workers(n_workers)

        if self.execution_mode == 'local':
            # Spawn local process pool
            self._submit_local(job_id, optimizer)
        else:
            # Submit to k8s
            self._submit_cloud(job_id, optimizer)

        return job_id

    def get_status(self, job_id: str) -> Dict:
        """Check job status (works for local or cloud)"""
        if self.execution_mode == 'local':
            return self._get_local_status(job_id)
        else:
            return self._get_cloud_status(job_id)

    def get_results(self, job_id: str) -> OptimizationResult:
        """Retrieve results (blocks if not complete)"""
        while True:
            status = self.get_status(job_id)
            if status['status'] == 'completed':
                return self._load_results(job_id)
            elif status['status'] == 'failed':
                raise OptimizationFailedError(status['error'])
            time.sleep(5)  # Poll every 5 seconds

    def get_results_async(self, job_id: str) -> Optional[OptimizationResult]:
        """Non-blocking result retrieval"""
        status = self.get_status(job_id)
        if status['status'] == 'completed':
            return self._load_results(job_id)
        return None

    def cancel_job(self, job_id: str) -> bool:
        """Cancel running job"""
        ...