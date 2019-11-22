import numpy as np
from marshmallow import fields
from marshmallow.validate import Length
from marshmallow_sqlalchemy import field_for
from sqlalchemy.ext.mutable import MutableDict

from src.db import db
from src.models.base import BaseModel, BaseSchema

INDEPENDENCE_TESTS = ["gaussCI", "disCI", "binCI"]


class Experiment(BaseModel):
    dataset_id = db.Column(db.Integer, db.ForeignKey('dataset.id'), nullable=False)
    dataset = db.relationship('Dataset')

    name = db.Column(db.String)
    algorithm_id = db.Column(db.Integer, db.ForeignKey('algorithm.id'))
    algorithm = db.relationship('Algorithm')

    description = db.Column(db.String)
    parameters = db.Column(MutableDict.as_mutable(db.JSON))

    @property
    def last_job(self):
        if len(self.jobs) == 0:
            return None
        return sorted(self.jobs, key=lambda x: x.start_time)[-1]

    @property
    def execution_time_statistics(self):
        execution_time_statistics = None
        execution_times = []
        for job in self.jobs:
            for result in job.results:
                if (result) and (result.execution_time is not None):
                    execution_times.append(result.execution_time)

        execution_times.sort()

        if execution_times:
            execution_time_statistics = {
                'min': min(execution_times),
                'max': max(execution_times),
                'mean': np.average(execution_times),
                'median': np.median(execution_times),
                'lower_quantile': np.quantile(execution_times, .25),
                'upper_quantile': np.quantile(execution_times, .75),
            }
        return execution_time_statistics

    @property
    def avg_execution_time(self):
        execution_time_sum = 0
        execution_time_count = 0
        for job in self.jobs:
            for result in job.results:
                if (result) and (result.execution_time is not None):
                    execution_time_sum = execution_time_sum + result.execution_time
                    execution_time_count = execution_time_count + 1

        if execution_time_count != 0:
            avg_execution_time = execution_time_sum / execution_time_count
            return avg_execution_time
        return 0.0


class ExperimentSchema(BaseSchema):
    name = field_for(Experiment, 'name', required=True, validate=Length(min=1))
    description = field_for(Experiment, 'description', required=False, allow_none=True, default='')
    algorithm = fields.Nested('AlgorithmSchema', dump_only=True)
    parameters = fields.Dict()
    last_job = fields.Nested('JobSchema', dump_only=True)
    execution_time_statistics = fields.Dict()

    class Meta(BaseSchema.Meta):
        model = Experiment
