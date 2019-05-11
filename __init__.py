__all__ = ['__version__']

# Main package modules:
from . import coreDef
__all__.extend( coreDef.__all__              )
from .coreDef import *
from . import CreateData
__all__.extend( CreateData.__all__           )
from .CreateData import *
from . import CreateTuningJobFiles
__all__.extend( CreateTuningJobFiles.__all__ )
from .CreateTuningJobFiles import *
from . import Neural
__all__.extend( Neural.__all__               )
from .Neural import *
from . import PreProc
__all__.extend( PreProc.__all__              )
from .PreProc import *
from . import TuningJob
__all__.extend( TuningJob.__all__            )
from .TuningJob import *
from . import TuningWrapper
__all__.extend( TuningWrapper.__all__        )
from .TuningWrapper import *
from . import CrossValid
__all__.extend( CrossValid.__all__           )
from .CrossValid import *
from . import CrossValidStat
__all__.extend( CrossValidStat.__all__       )
from .CrossValidStat import *
from . import SubsetGenerator
__all__.extend( SubsetGenerator.__all__      )
from .SubsetGenerator import *

# dataframe core
from . import dataframe
__all__.extend( dataframe.__all__            )
from dataframe import *
# parsers sub-package modules
from . import parsers
__all__.extend( parsers.__all__              )
from parsers import *
# plots sub-package modules
from . import monitoring
__all__.extend( monitoring.__all__           )
from monitoring import *

from RingerCore import masterLevel
masterLevel.mute( 'StoreGate'                )

__version__ = TuningToolsGit.tag
