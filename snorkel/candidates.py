from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from builtins import *
from future.utils import iteritems

from collections import defaultdict
from copy import deepcopy
from itertools import product
import re
from sqlalchemy.sql import select

from .models import Candidate
from .udf import UDF, UDFRunner

QUEUE_COLLECT_TIMEOUT = 5


class CandidateExtractor(UDFRunner):
    """
    An operator to extract Candidate objects from a Context.

    :param candidate_class: The type of relation to extract, defined using
                            :func:`snorkel.models.candidate_subclass <snorkel.models.candidate.candidate_subclass>`
    :param cspaces: one or list of :class:`CandidateSpace` objects, one for each relation argument. Defines space of
                    Contexts to consider
    :param matchers: one or list of :class:`snorkel.matchers.Matcher` objects, one for each relation argument. Only tuples of
                     Contexts for which each element is accepted by the corresponding Matcher will be returned as Candidates
    :param self_relations: Boolean indicating whether to extract Candidates that relate the same context.
                           Only applies to binary relations. Default is False.
    :param nested_relations: Boolean indicating whether to extract Candidates that relate one Context with another
                             that contains it. Only applies to binary relations. Default is False.
    :param symmetric_relations: Boolean indicating whether to extract symmetric Candidates, i.e., rel(A,B) and rel(B,A),
                                where A and B are Contexts. Only applies to binary relations. Default is False.
    """
    def __init__(self, candidate_class, cspaces, matchers, self_relations=False, nested_relations=False, symmetric_relations=False):
        super(CandidateExtractor, self).__init__(CandidateExtractorUDF,
                                                 candidate_class=candidate_class,
                                                 cspaces=cspaces,
                                                 matchers=matchers,
                                                 self_relations=self_relations,
                                                 nested_relations=nested_relations,
                                                 symmetric_relations=symmetric_relations)

    def apply(self, xs, split=0, **kwargs):
        super(CandidateExtractor, self).apply(xs, split=split, **kwargs)

    def clear(self, session, split, **kwargs):
        session.query(Candidate).filter(Candidate.split == split).delete()


class CandidateExtractorUDF(UDF):
    def __init__(self, candidate_class, cspaces, matchers, self_relations, nested_relations, symmetric_relations, **kwargs):
        self.candidate_class     = candidate_class
        # Note: isinstance is the way to check types -- not type(x) in [...]!
        self.candidate_spaces    = cspaces if isinstance(cspaces, (list, tuple)) else [cspaces]
        self.matchers            = matchers if isinstance(matchers, (list, tuple)) else [matchers]
        self.nested_relations    = nested_relations
        self.self_relations      = self_relations
        self.symmetric_relations = symmetric_relations

        super(CandidateExtractorUDF, self).__init__(**kwargs)

    def apply(self, context, clear, split, **kwargs):
       
        # Generates and persists candidates
        candidate_args = {'split': split}
           
        # Assemble candidate arguments
        for i, arg_name in enumerate(self.candidate_class.__argnames__):
            candidate_args[arg_name] = context

        # Add Candidate to session
        yield self.candidate_class(**candidate_args) 
