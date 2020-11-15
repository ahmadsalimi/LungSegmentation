"""
Returns class name -> class dictionary for each category
"""
from Auxiliary.RunsPhases.TrainPhase import TrainPhase
from Auxiliary.RunsPhases.EvalPhase import EvalPhase
from Auxiliary.RunsPhases.SavePhase import SavePhase
from Auxiliary.RunsPhases.ReportPhase import ReportPhase
from Auxiliary.RunsPhases.SaveSliceProbsPhase import SaveSliceProbsPhase
from Auxiliary.RunsPhases.SampleSliceReportPhase import SampleSliceReportPhase
from Auxiliary.ModelEvaluation.CovidEvaluators.BinarySuperInfoExtractor import BinarySuperInfoExtractor
from Auxiliary.ModelEvaluation.CovidEvaluators.BinaryCovidEvaluator import BinaryCovidEvaluator
from Auxiliary.ModelEvaluation.CovidEvaluators.PatchEvaluator import PatchEvaluator
from Auxiliary.ModelEvaluation.BinaryEvaluator import BinaryEvaluator
from Auxiliary.ModelEvaluation.CovidEvaluators.TertiaryCovidEvaluator import TertiaryCovidEvaluator
from Auxiliary.ModelEvaluation.CovidEvaluators.OvOEvaluator import OvOEvaluator
from Auxiliary.ModelEvaluation.CovidEvaluators.PatchInfectionSaver import PatchInfectionSaver
from Auxiliary.ModelEvaluation.MultiViewBinaryEvaluator import MultiViewBinaryEvaluator
from Auxiliary.ModelEvaluation.CovidEvaluators.MultiViewBinaryCovidEvaluator import MultiViewBinaryCovidEvaluator
from Auxiliary.ModelEvaluation.CovidEvaluators.MultiViewTertiaryEvaluator import MultiViewTertiaryEvaluator
from Auxiliary.RunsPhases.ModelOutputSavingPhase import ModelOutputSavingPhase


def get_phases_dict():
    """ Returns a dictionary from the name of phases to their constructor """
    phases_dict = dict()

    phases_dict['train'] = TrainPhase
    phases_dict['eval'] = EvalPhase
    phases_dict['save'] = SavePhase
    phases_dict['report'] = ReportPhase
    phases_dict['savesliceprobs'] = SaveSliceProbsPhase
    phases_dict['sampleslicereport'] = SampleSliceReportPhase
    phases_dict['savemodeloutput'] = ModelOutputSavingPhase

    return phases_dict


def get_evals_dict():
    """ Returns a dictionary from the name of evaluators to their constructor """
    evals_dict = dict()

    evals_dict['binary'] = BinaryEvaluator
    evals_dict['binarycovid'] = BinaryCovidEvaluator
    evals_dict['multiviewbinary'] = MultiViewBinaryEvaluator
    evals_dict['multiviewbinarycovid'] = MultiViewBinaryCovidEvaluator
    evals_dict['binarysuperinfo'] = BinarySuperInfoExtractor
    evals_dict['tertiary'] = TertiaryCovidEvaluator
    evals_dict['multiviewtertiary'] = MultiViewTertiaryEvaluator
    evals_dict['ovo'] = OvOEvaluator
    evals_dict['patch'] = PatchEvaluator
    evals_dict['patchinfectionsaver'] = PatchInfectionSaver

    return evals_dict
