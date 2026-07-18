import torch

from dist_net.data import gather_primary_incident


def test_primary_incident_gather_uses_stored_slot_not_first_active():
    feature = torch.zeros(2, 3, 12)
    feature[0, 0, 0] = 10.0
    feature[0, 2, 0] = 12.0
    feature[1, 0, 0] = 20.0
    feature[1, 1, 0] = 21.0
    relation = torch.zeros(2, 3, 4, 4)
    relation[0, 2] = 2.0
    relation[1, 1] = 1.0
    batch = {
        "incident_feat": feature,
        "incident_mask": torch.tensor(
            [[True, False, True], [True, True, False]]
        ),
        "rel_feat": relation,
        "primary_incident_slot": torch.tensor([2, 1]),
    }

    selected_feature, selected_relation, has_incident = (
        gather_primary_incident(batch, torch.device("cpu"))
    )

    torch.testing.assert_close(selected_feature[:, 0], torch.tensor([12.0, 21.0]))
    torch.testing.assert_close(selected_relation[:, 0, 0], torch.tensor([2.0, 1.0]))
    torch.testing.assert_close(has_incident, torch.ones(2))
