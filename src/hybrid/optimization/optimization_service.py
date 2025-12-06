from datetime import time, datetime
from typing import Optional, Dict

from src.hybrid.optimization import OptimizerType
from src.hybrid.optimization.optimization_result import OptimizationResult
from src.hybrid.optimization.optimizer_factory import OptimizerFactory

class OptimizationFailedError(Exception):
    """Raised when optimization job fails"""
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)

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
                            strategy,
                            optimizer_type: OptimizerType,
                            n_workers: int = 16) -> str:
        """Submit optimization job (async, returns immediately)"""
        job_id = self._generate_job_id()

        # Use self.config instead of undefined config
        optimizer = OptimizerFactory.create_optimizer(optimizer_type, self.config, strategy)

        if self.execution_mode == 'local':
            # Spawn local process pool
            self._submit_local(job_id, optimizer, n_workers)
        else:
            # Submit to k8s
            self._submit_cloud(job_id, optimizer, n_workers)

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

    def _generate_job_id(self) -> str:
        """Generate unique job ID"""
        import uuid
        from datetime import datetime
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        return f"opt_{timestamp}_{uuid.uuid4().hex[:8]}"

    def _submit_local(self, job_id: str, optimizer, n_workers: int):
        """Submit to local process pool"""
        from multiprocessing import Process

        # Store job info
        self.jobs[job_id] = {
            'status': 'running',
            'optimizer': optimizer,
            'n_workers': n_workers,
            'started_at': datetime.now().isoformat()
        }

        # Start optimization in separate process
        process = Process(target=self._run_optimization_process, args=(job_id, optimizer, n_workers))
        process.start()

        # Store process for tracking
        self.jobs[job_id]['process'] = process

    def _submit_cloud(self, job_id: str, optimizer, n_workers: int):
        """Submit to k8s cluster"""
        # TODO: Implement cloud execution
        raise NotImplementedError("Cloud execution not yet implemented")

    def _run_optimization_process(self, job_id: str, optimizer, n_workers: int):
        """Worker process that runs the actual optimization"""
        try:
            # Run optimization
            results = optimizer.run_optimization(n_workers=n_workers)

            # Save results
            self._save_results(job_id, results)

            # Update status
            self.jobs[job_id]['status'] = 'completed'
            self.jobs[job_id]['completed_at'] = datetime.now().isoformat()

        except Exception as e:
            self.jobs[job_id]['status'] = 'failed'
            self.jobs[job_id]['error'] = str(e)
            self.jobs[job_id]['failed_at'] = datetime.now().isoformat()

    def _save_results(self, job_id: str, results):
        """Save optimization results to disk"""
        import json
        from pathlib import Path

        results_dir = Path('optimization_results')
        results_dir.mkdir(exist_ok=True)

        results_file = results_dir / f'{job_id}.json'
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)