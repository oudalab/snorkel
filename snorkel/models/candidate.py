from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from builtins import *

from sqlalchemy import (
    Column, String, Integer, Float, Boolean, ForeignKey, UniqueConstraint,
    MetaData
)
from sqlalchemy.orm import relationship, backref
from functools import partial

from .meta import SnorkelBase
from ..models import snorkel_engine
from ..utils import camel_to_under


class Candidate(SnorkelBase):
    """
    An abstract candidate relation.

    New relation types should be defined by calling candidate_subclass(),
    **not** subclassing this class directly.
    """
    __tablename__ = 'candidate'
    id          = Column(Integer, primary_key=True)
    split       = Column(Integer, nullable=False, default=0, index=True)

    __mapper_args__ = {
        'polymorphic_identity': 'candidate',
    }

    # __table_args__ = {"extend_existing" : True}

    def __len__(self):
        return len(self.__argnames__)

# This global dictionary contains all classes that have been declared in this Python environment, so
# that candidate_subclass() can return a class if it already exists and is identical in specification
# to the requested class
candidate_subclasses = {}

def candidate_subclass(class_name, args, table_name=None, cardinality=None,
    values=None):
    """
    Creates and returns a Candidate subclass with provided argument names, 
    which are Context type. Creates the table in DB if does not exist yet.

    Import using:

    .. code-block:: python

        from snorkel.models import candidate_subclass

    :param class_name: The name of the class, should be "camel case" e.g. 
        NewCandidate
    :param args: A list of names of consituent arguments, which refer to the 
        Contexts--representing mentions--that comprise the candidate
    :param table_name: The name of the corresponding table in DB; if not 
        provided, is converted from camel case by default, e.g. new_candidate
    :param cardinality: The cardinality of the variable corresponding to the
        Candidate. By default is 2 i.e. is a binary value, e.g. is or is not
        a true mention.
    """
    if table_name is None:
        table_name = camel_to_under(class_name)

    # If cardinality and values are None, default to binary classification
    if cardinality is None and values is None:
        values = [True, False]
        cardinality = 2
    
    # Else use values if present, and validate proper input
    elif values is not None:
        if cardinality is not None and len(values) != cardinality:
            raise ValueError("Number of values must match cardinality.")
        if None in values:
            raise ValueError("`None` is a protected value.")
        # Note that bools are instances of ints in Python...
        if any([isinstance(v, int) and not isinstance(v, bool) for v in values]):
            raise ValueError("Default usage of values is consecutive integers. Leave values unset if attempting to define values as integers.")
        cardinality = len(values)

    # If cardinality is specified but not values, fill in with ints
    elif cardinality is not None:
        values = list(range(cardinality))

    class_spec = (args, table_name, cardinality, values)
    if class_name in candidate_subclasses:
        if class_spec == candidate_subclasses[class_name][1]:
            return candidate_subclasses[class_name][0]
        else:
            raise ValueError('Candidate subclass ' + class_name + ' already exists in memory with incompatible ' +
                             'specification: ' + str(candidate_subclasses[class_name][1]))
    else:
        # Set the class attributes == the columns in the database
        class_attribs = {

            # Declares name for storage table
            '__tablename__' : table_name,

            # Connects candidate_subclass records to generic Candidate records
            'id' : Column(
                Integer,
                ForeignKey('candidate.id', ondelete='CASCADE'),
                primary_key=True
            ),

            # Store values & cardinality information in the class only
            'values' : values,
            'cardinality' : cardinality,

            # Polymorphism information for SQLAlchemy
            '__mapper_args__' : {'polymorphic_identity': table_name},

            # Helper method to get argument names
            '__argnames__' : args,
        }

        # Create named arguments, i.e. the entity mentions comprising the relation
        # mention
        # For each entity mention: id, cid ("canonical id"), and pointer to Context
        for arg in args:

            # Primary arguments are constituent Contexts, and their ids
            class_attribs[arg] = Column(String)

        # Create class
        C = type(class_name, (Candidate,), class_attribs)

        # Create table in DB
        if not snorkel_engine.dialect.has_table(snorkel_engine, table_name):
            C.__table__.create(bind=snorkel_engine)

        candidate_subclasses[class_name] = C, class_spec

        return C


class Marginal(SnorkelBase):
    """
    A marginal probability corresponding to a (Candidate, value) pair.

    Represents:

        P(candidate = value) = probability

    @training: If True, this is a training marginal; otherwise is end prediction
    """
    __tablename__ = 'marginal'
    id           = Column(Integer, primary_key=True)
    candidate_id = Column(Integer, 
                        ForeignKey('candidate.id', ondelete='CASCADE'))
    training     = Column(Boolean, default=True)
    value        = Column(Integer, nullable=False, default=1)
    probability  = Column(Float, nullable=False, default=0.0)
    
    __table_args__ = (
        UniqueConstraint(candidate_id, training, value),
    )

    def __repr__(self):
        label = "Training" if self.training else "Predicted"
        return "<%s Marginal: P(%s == %s) = %s>" % \
            (label, self.candidate_id, self.value, self.probability)
