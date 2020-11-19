"""experiment job refactoring

Revision ID: 7cfcdc8ab2e3
Revises: 6c819ffe83d6
Create Date: 2020-11-13 12:53:26.727132

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '7cfcdc8ab2e3'
down_revision = '6c819ffe83d6'
branch_labels = None
depends_on = None


def upgrade():
    op.create_table('experiment_job',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('experiment_id', sa.Integer(), nullable=False),
        sa.Column('parallel', sa.Boolean(), nullable=True),
        sa.Column('enforce_cpus', sa.Boolean(), default=True),
        sa.Column('gpus', sa.Integer(), nullable=True),
        sa.ForeignKeyConstraint(['experiment_id'], ['experiment.id'], ),
        sa.ForeignKeyConstraint(['id'], ['job.id'], ),
        sa.PrimaryKeyConstraint('id')
    )

    op.execute('INSERT INTO experiment_job(id, experiment_id, parallel, enforce_cpus, gpus) '\
               'SELECT id, experiment_id, parallel, enforce_cpus, gpus FROM job')
    op.drop_column('job', 'experiment_id')
    op.drop_column('job', 'parallel')
    op.drop_column('job', 'enforce_cpus')
    op.drop_column('job', 'gpus')

    op.add_column('job', sa.Column('type', sa.String(), nullable=True))
    op.execute('UPDATE job SET type=\'experiment_job\'')
    op.alter_column('job', 'type', nullable=False)

def downgrade():
    op.add_column('job', sa.Column('experiment_id', sa.Integer(), nullable=True))
    op.add_column('job', sa.Column('parallel', sa.Boolean(), nullable=True))
    op.add_column('job', sa.Column('enforce_cpus', sa.Boolean(), default=True))
    op.add_column('job', sa.Column('gpus', sa.Integer(), nullable=True))
    op.drop_column('job', 'type')

    op.execute('UPDATE job SET '\
                'experiment_id=experiment_job.experiment_id, '\
                'parallel=experiment_job.parallel, '\
                'enforce_cpus=experiment_job.enforce_cpus, '\
                'gpus=experiment_job.gpus '\
            'FROM (SELECT id, experiment_id, parallel, enforce_cpus, gpus FROM experiment_job) AS experiment_job '\
            'WHERE job.id = experiment_job.id')
    op.alter_column('job', 'experiment_id', nullable=False)

    op.drop_table('experiment_job')
