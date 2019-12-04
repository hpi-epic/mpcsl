"""Added waiting JobStatus

Revision ID: 4b3174c88880
Revises: bde7db78a4fb
Create Date: 2019-11-21 22:27:57.061300

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '4b3174c88880'
down_revision = 'bde7db78a4fb'
branch_labels = None
depends_on = None


old_options = ('running', 'done', 'error', 'cancelled', 'hidden')
new_options = sorted(old_options + ('waiting',))

old_type = sa.Enum(*old_options, name='jobstatus')
new_type = sa.Enum(*new_options, name='jobstatus')
tmp_type = sa.Enum(*new_options, name='_jobstatus')

tcr = sa.sql.table('job',
                   sa.Column('status', new_type, nullable=False))


def upgrade():
    # Create a tempoary "_jobstatus" type, convert and drop the "old" type
    tmp_type.create(op.get_bind(), checkfirst=False)
    op.execute('ALTER TABLE job ALTER COLUMN status TYPE _jobstatus'
               ' USING status::text::_jobstatus')
    old_type.drop(op.get_bind(), checkfirst=False)
    # Create and convert to the "new" jobstatus type
    new_type.create(op.get_bind(), checkfirst=False)
    op.execute('ALTER TABLE job ALTER COLUMN status TYPE jobstatus'
               ' USING status::text::jobstatus')
    tmp_type.drop(op.get_bind(), checkfirst=False)


def downgrade():
    # Convert 'output_limit_exceeded' jobstatus into 'timed_out'
    op.execute(tcr.update().where(tcr.c.status==u'waiting')
               .values(status='running'))
    # Create a tempoary "_jobstatus" type, convert and drop the "new" type
    tmp_type.create(op.get_bind(), checkfirst=False)
    op.execute('ALTER TABLE job ALTER COLUMN status TYPE _jobstatus'
               ' USING status::text::_jobstatus')
    new_type.drop(op.get_bind(), checkfirst=False)
    # Create and convert to the "old" jobstatus type
    old_type.create(op.get_bind(), checkfirst=False)
    op.execute('ALTER TABLE job ALTER COLUMN status TYPE jobstatus'
               ' USING status::text::jobstatus')
    tmp_type.drop(op.get_bind(), checkfirst=False)
